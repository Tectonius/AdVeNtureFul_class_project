"""Turn tile evidence into fixed-size CNN examples, one per candidate.

An example is a (n_features, width) matrix: `width` reference positions
centred on the candidate (the candidate is column width // 2), and for each
position the features below. All features are scaled to roughly [0, 1] so
examples from 35x and 48x data look alike.

    ref_A .. ref_T        one-hot reference base (all 0 for N / off-contig)
    depth_norm            depth / mean depth of the tile
    frac_A .. frac_N      fraction of depth showing each base
    frac_del_span         fraction of depth with this base deleted
    frac_ins_after        fraction of depth with an insertion after this base
    frac_del_after        fraction of depth with a deletion after this base
    rev_frac_depth        fraction of depth from reverse-strand reads
    rev_frac_nonref       ... of the non-reference bases (0.5 if none)
    rev_frac_ins          ... of the insertions (0.5 if none)
    rev_frac_del          ... of the deletions (0.5 if none)
    mean_bq               mean base quality / 60, capped at 1
    mean_mapq             mean mapping quality / 60, capped at 1
    mean_ins_len          mean insertion length here / 50, capped at 1
    mean_del_len          mean deletion length here / 50, capped at 1

Positions outside the contig are all zeros.
"""

import numpy as np

from .evidence import (A, BQ_SUM, DEL_AFTER, DEL_LEN_SUM, DEL_SPAN, INS_AFTER,
                       INS_LEN_SUM, MAPQ_SUM, N, T)

FEATURE_NAMES = [
    "ref_A", "ref_C", "ref_G", "ref_T",
    "depth_norm",
    "frac_A", "frac_C", "frac_G", "frac_T", "frac_N",
    "frac_del_span", "frac_ins_after", "frac_del_after",
    "rev_frac_depth", "rev_frac_nonref", "rev_frac_ins", "rev_frac_del",
    "mean_bq", "mean_mapq", "mean_ins_len", "mean_del_len",
]
N_FEATURES = len(FEATURE_NAMES)
EXAMPLE_DTYPE = np.float16


def _ratio(num, den, empty=0.0):
    """num / den elementwise, `empty` where den == 0."""
    out = np.full(num.shape, empty, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0)
    return out


def feature_matrix(evidence):
    """(N_FEATURES, tile_len) float32 features for every position of the tile."""
    counts, rev = evidence.counts, evidence.rev_counts
    depth = evidence.depth
    base_depth = counts[A:N + 1].sum(axis=0)
    rev_depth = rev[A:N + 1].sum(axis=0) + rev[DEL_SPAN]
    covered = depth[depth > 0]
    mean_depth = covered.mean() if len(covered) else 1.0

    ref_onehot = np.zeros((4, len(depth)), dtype=np.float32)
    known = evidence.ref_codes < N
    ref_onehot[evidence.ref_codes[known], np.flatnonzero(known)] = 1.0

    nonref = base_depth - counts[N] - evidence.ref_counts
    rev_nonref = rev[A:N + 1].sum(axis=0) - rev[N] - evidence.rev_ref_counts

    rows = [
        *ref_onehot,
        depth / mean_depth,
        *(_ratio(counts[b], depth) for b in range(A, N + 1)),
        _ratio(counts[DEL_SPAN], depth),
        _ratio(counts[INS_AFTER], depth),
        _ratio(counts[DEL_AFTER], depth),
        _ratio(rev_depth, depth),
        _ratio(rev_nonref, nonref, empty=0.5),
        _ratio(rev[INS_AFTER], counts[INS_AFTER], empty=0.5),
        _ratio(rev[DEL_AFTER], counts[DEL_AFTER], empty=0.5),
        np.minimum(_ratio(evidence.sums[BQ_SUM], base_depth) / 60, 1),
        np.minimum(_ratio(evidence.sums[MAPQ_SUM], evidence.mapq_depth) / 60, 1),
        np.minimum(_ratio(evidence.sums[INS_LEN_SUM], counts[INS_AFTER]) / 50, 1),
        np.minimum(_ratio(evidence.sums[DEL_LEN_SUM], counts[DEL_AFTER]) / 50, 1),
    ]
    return np.stack(rows).astype(np.float32)


def build_examples(evidence, positions, width):
    """Examples for 1-based candidate `positions` inside the evidence tile.

    Returns an array of shape (len(positions), N_FEATURES, width).
    """
    half = width // 2
    # Transposed to (position, feature) so each example is one contiguous
    # slice; zero-padded so windows near the contig edge stay full width.
    features = feature_matrix(evidence).T.astype(EXAMPLE_DTYPE)
    padded = np.pad(features, ((half, width - half), (0, 0)))
    centers = np.asarray(positions) - 1 - evidence.start + half

    out = np.empty((len(centers), N_FEATURES, width), dtype=EXAMPLE_DTYPE)
    for i, c in enumerate(centers):
        out[i] = padded[c - half:c + width - half].T
    return out


def load_examples(out_dir):
    """(examples memmap, metadata DataFrame) from a find_candidates run.

    The array is memory-mapped, so e.g. a PyTorch Dataset can index single
    rows without loading the whole file.
    """
    import os
    import pandas as pd
    arr = np.load(os.path.join(out_dir, "examples.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(out_dir, "examples.tsv"), sep="\t")
    return arr, meta
