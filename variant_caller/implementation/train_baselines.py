#!/usr/bin/env python3
"""Baseline classifiers on the candidate's centre column only.

    python train_baselines.py --examples out/ [--results baselines.tsv]
        [--val-chroms chr21 chr22 --test-chroms chr20]

Each example is reduced to its centre-column features (the candidate
position itself, no neighbouring context). If the CNN cannot beat these,
it is not using the context.

Models:
    threshold_rule        one cutoff on the best alt fraction (SNP/ins/del),
                          tuned on train; what a hand-written caller would do
    gaussian_nb           Gaussian naive Bayes
    logistic_regression   standardized features, L2
    gradient_boosting     HistGradientBoostingClassifier

Split: with --test-chroms, whole chromosomes are held out (GIAB convention:
test chr20, validate chr21-22). With one contig (e.g. bacteria), contiguous
position blocks are used instead: first 60% train, next 20% val, last 20% test,
so neighbouring, correlated candidates never straddle train and test.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from candidates.examples import FEATURE_NAMES, load_examples

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_center_features(out_dir):
    """(X centre-column features float32, y labels, metadata) from a run."""
    examples, meta = load_examples(out_dir)
    center = examples.shape[2] // 2
    X = np.asarray(examples[:, :, center], dtype=np.float32)
    if (meta["label"] < 0).any():
        raise SystemExit("examples are unlabelled; rerun find_candidates.py with --truth")
    return X, meta["label"].to_numpy(), meta


def split_indices(meta, val_chroms=None, test_chroms=None):
    """{'train'|'val'|'test': row indices}, by chromosome or by position block."""
    if test_chroms:
        chrom = meta["chrom"].astype(str)
        test = chrom.isin(test_chroms).to_numpy()
        val = chrom.isin(val_chroms or []).to_numpy()
        train = ~(test | val)
    else:
        # Rank of each candidate along the genome (contig order, then position).
        order = meta.reset_index().sort_values(["chrom", "pos"])["index"].to_numpy()
        rank = np.empty(len(meta), dtype=np.int64)
        rank[order] = np.arange(len(meta))
        frac = rank / len(meta)
        train, val, test = frac < 0.6, (frac >= 0.6) & (frac < 0.8), frac >= 0.8
    return {name: np.flatnonzero(mask) for name, mask in
            (("train", train), ("val", val), ("test", test))}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def best_alt_fraction(X):
    """Largest non-reference signal at the centre: alt base, insertion or deletion."""
    return X[:, F["best_alt_fraction"]]


class ThresholdRule:
    """Predict 'variant' when best_alt_fraction >= a cutoff chosen for train F1."""

    def fit(self, X, y):
        score = best_alt_fraction(X)
        cutoffs = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(y > 0, score >= c) for c in cutoffs]
        self.cutoff_ = cutoffs[int(np.argmax(f1s))]
        return self

    def predict(self, X):
        return (best_alt_fraction(X) >= self.cutoff_).astype(int)

    def predict_proba(self, X):
        s = best_alt_fraction(X)
        return np.column_stack([1 - s, s])


def make_models(seed=0):
    return {
        "threshold_rule": ThresholdRule(),
        "gaussian_nb": GaussianNB(),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1, random_state=seed),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def score(y, pred, prob):
    """Metrics for one set of predictions ("positive" = any variant)."""
    binary = len(np.unique(y)) <= 2 and prob.shape[1] == 2
    is_var, pred_var = y > 0, pred > 0
    out = {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(is_var, pred_var, zero_division=0),
        "recall": recall_score(is_var, pred_var, zero_division=0),
        "f1": f1_score(is_var, pred_var, zero_division=0),
    }
    p_var = prob[:, 1] if binary else 1 - prob[:, 0]
    if is_var.any() and (~is_var).any():
        out["roc_auc"] = roc_auc_score(is_var, p_var)
        out["avg_precision"] = average_precision_score(is_var, p_var)
    if not binary:
        out["genotype_macro_f1"] = f1_score(y, pred, average="macro")
    return out


def candidate_group(types):
    """'SNP' for SNP-only candidates, 'INDEL' for anything with an indel."""
    return np.where(types.str.contains("INS|DEL"), "INDEL", "SNP")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--examples", required=True, help="output folder of find_candidates.py")
    p.add_argument("--val-chroms", nargs="*")
    p.add_argument("--test-chroms", nargs="*")
    p.add_argument("--results", help="write the metrics table here (TSV)")
    p.add_argument("--predictions", help="write per-example test predictions here (TSV)")
    args = p.parse_args()

    X, y, meta = load_center_features(args.examples)
    splits = split_indices(meta, args.val_chroms, args.test_chroms)
    groups = candidate_group(meta["type"])
    print(f"{len(y)} examples, {X.shape[1]} centre features; label counts {dict(zip(*np.unique(y, return_counts=True)))}")
    for name, idx in splits.items():
        print(f"  {name:<5} n={len(idx):>7}  positive rate={np.mean(y[idx] > 0):.3f}")

    rows = []
    tr, te = splits["train"], splits["test"]
    test_pred = meta.iloc[te][["chrom", "pos", "ref", "type", "label", "truth_kind"]].copy()
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(X[tr], y[tr])
        fit_s = time.time() - t0
        for split in ("val", "test"):
            idx = splits[split]
            pred, prob = model.predict(X[idx]), model.predict_proba(X[idx])
            if split == "test":
                test_pred[f"p_{name}"] = prob[:, 1] if prob.shape[1] == 2 else 1 - prob[:, 0]
                test_pred[f"pred_{name}"] = pred
            for group in ("ALL", "SNP", "INDEL"):
                keep = np.ones(len(idx), bool) if group == "ALL" else groups[idx] == group
                rows.append({"model": name, "split": split, "candidates": group,
                             "fit_s": round(fit_s, 1), **score(y[idx][keep], pred[keep], prob[keep])})
        if name == "threshold_rule":
            print(f"  threshold_rule cutoff = {model.cutoff_:.2f}")

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for split in ("val", "test"):
        print(f"\n=== {split} ===")
        print(table[table.split == split].drop(columns="split").round(4).to_string(index=False))

    # Which features the linear model leans on (standardized coefficients).
    lr = make_models()["logistic_regression"].fit(X[tr], y[tr])
    coef = pd.Series(lr[-1].coef_[0], index=FEATURE_NAMES).sort_values(key=np.abs, ascending=False)
    print("\nlogistic regression, largest standardized coefficients:")
    print(coef.head(8).round(3).to_string())

    if args.predictions:
        test_pred.to_csv(args.predictions, sep="\t", index=False)
        print(f"test predictions -> {args.predictions}")
    if args.results:
        table.to_csv(args.results, sep="\t", index=False)
        print(f"\nmetrics -> {args.results}")


if __name__ == "__main__":
    main()
