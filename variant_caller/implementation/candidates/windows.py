"""Group nearby candidates into windows.

Candidates on the same contig that are less than `max_gap` bp apart are
chained into one window, so clustered variants (e.g. a SNP next to an indel)
are handed to the next stage together.
"""

import numpy as np
import pandas as pd

WINDOW_COLUMNS = ["chrom", "start", "end", "n_candidates", "types"]


def merge_into_windows(candidates, max_gap):
    """Candidates DataFrame (1-based pos) -> BED-style windows DataFrame.

    Windows are 0-based half-open [start, end) spanning the first to the last
    candidate base in the cluster.
    """
    rows = []
    for chrom, group in candidates.groupby("chrom", sort=False):
        group = group.sort_values("pos")
        positions = group["pos"].to_numpy()
        types = group["type"].to_numpy()

        # A new window begins wherever the gap to the previous candidate is too big.
        breaks = [0] + [i for i in range(1, len(positions))
                        if positions[i] - positions[i - 1] >= max_gap] + [len(positions)]

        for first, stop in zip(breaks[:-1], breaks[1:]):
            window_types = sorted({t for ts in types[first:stop] for t in ts.split(",")})
            rows.append((chrom, positions[first] - 1, positions[stop - 1],
                         stop - first, ",".join(window_types)))

    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def assign_windows(table, windows):
    """Row index into `windows` for each (chrom, pos) row of `table`, or -1."""
    result = np.full(len(table), -1, dtype=np.int64)
    for chrom, win in windows.groupby("chrom", sort=False):
        rows = np.flatnonzero((table["chrom"] == chrom).to_numpy())
        pos0 = table["pos"].to_numpy()[rows] - 1
        i = np.searchsorted(win["start"].to_numpy(), pos0, side="right") - 1
        inside = (i >= 0) & (pos0 < win["end"].to_numpy()[np.maximum(i, 0)])
        result[rows[inside]] = win.index.to_numpy()[i[inside]]
    return result
