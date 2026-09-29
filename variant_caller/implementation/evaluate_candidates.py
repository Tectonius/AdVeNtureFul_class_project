#!/usr/bin/env python3
"""Measure how many truth variants the candidate step recovers, and why not.

    python evaluate_candidates.py --out-dir out/ --truth truth.vcf.gz \
        [--bed confident.bed] [--bam reads.bam --ref ref.fasta --threads 8]

Reads out/candidates.tsv, out/windows.bed and out/config.json.

A truth variant counts as "found" if a candidate sits on its VCF POS (the
anchor base for indels, the base itself for SNPs), and as "in a window" if
its POS falls inside any window. Truth records are first left-normalized
against the reference (like `bcftools norm -f`) so POS means the same thing
on both sides. Recall is what matters here: anything this
step misses, the CNN never gets a chance to call.

With --bam/--ref, every missed variant is diagnosed from the read evidence at
its POS and written to out/missed.tsv with one of these reasons:

    no_coverage    no reads cover the position (after read filters)
    no_support     reads cover it, but none show this kind of event here
    one_read       support below min_alt_count
    low_fraction   enough reads, but support / depth below the fraction cutoff
    represented_nearby  (COMPLEX) reads clearly disagree with the reference
                   here, but the aligner anchored the event on nearby bases
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysam

from candidates import CandidateConfig, collect_tile_evidence
from candidates.evidence import BASE_CODE, DEL_AFTER, INS_AFTER
from candidates.truth import in_intervals, load_bed, normalize, truth_kind

NEARBY_BP = 10  # "a candidate was close" radius for misses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_distance(sorted_positions, pos):
    if len(sorted_positions) == 0:
        return np.inf
    i = np.searchsorted(sorted_positions, pos)
    near = sorted_positions[max(i - 1, 0):i + 1]
    return int(np.abs(near - pos).min())


# ---------------------------------------------------------------------------
# Diagnosing misses from read evidence
# ---------------------------------------------------------------------------

_bam = _fasta = _config = None


def _init_worker(bam_path, ref_path, config):
    global _bam, _fasta, _config
    _bam = pysam.AlignmentFile(bam_path, "rb")
    _fasta = pysam.FastaFile(ref_path)
    _config = config


def _diagnose_tile(job):
    """job = (contig, tile_start, tile_end, [miss dicts]) -> miss dicts with evidence."""
    contig, start, end, misses = job
    ev = collect_tile_evidence(_bam, _fasta, contig, start, end, _config)
    depth = ev.depth
    for m in misses:
        i = m["pos"] - 1 - start
        m["depth"] = int(depth[i])
        if m["kind"] == "SNP":
            m["support"] = int(ev.counts[BASE_CODE[ord(m["alt"][0])], i])
        elif m["kind"] == "INS":
            m["support"] = int(ev.counts[INS_AFTER, i])
        elif m["kind"] == "DEL":
            m["support"] = int(ev.counts[DEL_AFTER, i])
        else:  # complex: best of any event at this base
            m["support"] = int(max(ev.counts[INS_AFTER, i], ev.counts[DEL_AFTER, i],
                                   depth[i] - ev.ref_counts[i]))
        m["reason"] = miss_reason(m, _config)
    return misses


def miss_reason(m, config):
    if m["depth"] == 0:
        return "no_coverage"
    if m["support"] == 0:
        return "no_support"
    if m["support"] < config.min_alt_count:
        return "one_read"
    min_fraction = config.min_snp_fraction if m["kind"] == "SNP" else config.min_indel_fraction
    if m["support"] < min_fraction * m["depth"]:
        return "low_fraction"
    # Only reachable for COMPLEX: plenty of non-reference reads at POS, but the
    # aligner wrote the event as SNPs/indels anchored on neighbouring bases.
    return "represented_nearby"


def diagnose(misses, bam_path, ref_path, config, threads):
    jobs = defaultdict(list)
    for m in misses:
        tile = (m["pos"] - 1) // config.tile_size * config.tile_size
        jobs[(m["chrom"], tile)].append(m)
    with pysam.FastaFile(ref_path) as fasta:
        work = [(c, s, min(s + config.tile_size, fasta.get_reference_length(c)), ms)
                for (c, s), ms in jobs.items()]
    with Pool(threads, initializer=_init_worker, initargs=(bam_path, ref_path, config)) as pool:
        return [m for part in pool.imap(_diagnose_tile, work) for m in part]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(title, rows, columns):
    print(f"\n{title}")
    print(pd.DataFrame(rows, columns=columns).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="output folder of find_candidates.py")
    p.add_argument("--truth", required=True)
    p.add_argument("--bed", help="restrict to confident regions")
    p.add_argument("--bam", help="with --ref: diagnose why each miss was missed")
    p.add_argument("--ref")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--truth-ref", help="FASTA used to left-normalize truth (default: --ref)")
    p.add_argument("--no-normalize", action="store_true", help="use truth POS as written")
    args = p.parse_args()
    norm_ref = args.truth_ref or args.ref
    if not args.no_normalize and not norm_ref:
        p.error("need --ref (or --truth-ref) to normalize truth, or pass --no-normalize")

    cands = pd.read_csv(os.path.join(args.out_dir, "candidates.tsv"), sep="\t")
    windows = load_bed(os.path.join(args.out_dir, "windows.bed"))
    with open(os.path.join(args.out_dir, "config.json")) as fh:
        config = CandidateConfig(**json.load(fh))
    confident = load_bed(args.bed) if args.bed else None

    cand_sites = set(zip(cands["chrom"], cands["pos"]))
    cand_positions = {c: np.sort(g["pos"].to_numpy()) for c, g in cands.groupby("chrom")}

    total, found, in_window = Counter(), Counter(), Counter()
    true_sites, misses, n_moved = set(), [], 0
    fasta = None if args.no_normalize else pysam.FastaFile(norm_ref)
    with pysam.VariantFile(args.truth) as vcf:
        for contig in sorted(cand_positions):
            if contig not in vcf.header.contigs:
                continue
            for rec in vcf.fetch(contig):
                if confident and not in_intervals(confident, rec.chrom, rec.pos - 1):
                    continue
                pos, ref, alt = rec.pos, rec.ref, rec.alts[0]
                if fasta:
                    pos, ref, alt = normalize(fasta, rec.chrom, pos, ref, alt)
                    n_moved += pos != rec.pos
                kind = truth_kind(ref, alt)
                site = (rec.chrom, pos)
                total[kind] += 1
                true_sites.add(site)
                win = in_intervals(windows, rec.chrom, pos - 1)
                in_window[kind] += win
                if site in cand_sites:
                    found[kind] += 1
                else:
                    misses.append({
                        "chrom": rec.chrom, "pos": pos, "ref": ref, "alt": alt,
                        "vcf_pos": rec.pos, "kind": kind, "in_window": win,
                        "nearest_candidate": nearest_distance(cand_positions[contig], pos),
                    })
    if fasta:
        print(f"truth records moved by normalization: {n_moved}")

    kinds = sorted(total)
    rows = [(k, total[k], found[k], total[k] - found[k], found[k] / total[k],
             in_window[k], total[k] - in_window[k], in_window[k] / total[k]) for k in kinds]
    t, f, w = sum(total.values()), sum(found.values()), sum(in_window.values())
    rows.append(("ALL", t, f, t - f, f / max(t, 1), w, t - w, w / max(t, 1)))
    print_table("Recall of truth variants", [tuple(round(x, 4) if isinstance(x, float) else x for x in r)
                                             for r in rows],
                ["type", "truth", "found", "missed", "recall", "in_window", "not_in_window", "win_recall"])

    if confident:
        cand_sites = {s for s in cand_sites if in_intervals(confident, s[0], s[1] - 1)}
    true_cands = len(cand_sites & true_sites)
    print(f"\ncandidates: {len(cand_sites)}  at a truth POS: {true_cands} "
          f"({true_cands / max(len(cand_sites), 1):.3f})  windows: {sum(len(v) for v in windows.values())}")

    if not misses:
        return
    if args.bam and args.ref:
        misses = diagnose(misses, args.bam, args.ref, config, args.threads)
    missed = pd.DataFrame(misses)
    missed["candidate_within_10bp"] = missed["nearest_candidate"] <= NEARBY_BP

    group = ["kind", "reason"] if "reason" in missed else ["kind"]
    summary = missed.groupby(group).agg(
        missed=("pos", "size"),
        cand_within_10bp=("candidate_within_10bp", "sum"),
        in_window=("in_window", "sum"),
    ).reset_index()
    if "reason" in missed:
        extra = missed.groupby(group).agg(median_depth=("depth", "median"),
                                          median_support=("support", "median")).reset_index()
        summary = summary.merge(extra, on=group)
    print_table("Missed truth variants (no candidate at POS)", summary, summary.columns)

    missed_path = os.path.join(args.out_dir, "missed.tsv")
    missed.sort_values(["chrom", "pos"]).to_csv(missed_path, sep="\t", index=False)
    print(f"\nper-variant detail -> {missed_path}")


if __name__ == "__main__":
    main()
