"""Turn per-position evidence into a table of candidate variant positions.

A position is a candidate if *any* of these holds:

    SNP  the most common non-reference base has >= min_alt_count reads
         and makes up >= min_snp_fraction of the depth
    INS  reads with an insertion after this base pass the indel thresholds
    DEL  reads with a deletion after this base pass the indel thresholds

Every candidate row also carries the reference-supporting count, so the next
stage (and a human reading the table) can see the evidence both ways, and
the most common insertion / deletion length among the supporting reads.
"""

import numpy as np
import pandas as pd

from .evidence import BASES, DEL_AFTER, INS_AFTER, N

CANDIDATE_COLUMNS = [
    "chrom", "pos", "ref", "type", "depth", "ref_count",
    "alt_base", "snp_count", "ins_count", "del_count", "ins_len", "del_len",
]


def _passes(alt_count, depth, min_count, min_fraction):
    return (alt_count >= min_count) & (alt_count >= min_fraction * depth)


def modal_indel_lengths(indel_events, offsets, is_insertion):
    """Most common indel length at each offset (ties -> longer); 0 if none."""
    if len(offsets) == 0 or len(indel_events) == 0:
        return np.zeros(len(offsets), dtype=np.int64)
    ev = pd.DataFrame(indel_events, columns=["offset", "is_ins", "length"])
    ev = ev[(ev.is_ins == is_insertion) & ev.offset.isin(offsets)]
    modal = (ev.groupby(["offset", "length"]).size().rename("n").reset_index()
               .sort_values(["n", "length"], ascending=False)
               .drop_duplicates("offset").set_index("offset")["length"])
    return modal.reindex(offsets, fill_value=0).to_numpy()


def find_candidates(evidence, config, core_start=None, core_end=None):
    """Return a DataFrame of candidates for one TileEvidence.

    Only positions in [core_start, core_end) are reported (default: the whole
    tile); the evidence may extend beyond that to give examples context.

    `pos` is 1-based (VCF / IGV convention); indel candidates sit on the
    base before the event, exactly like a VCF indel record's POS.
    """
    core_start = evidence.start if core_start is None else core_start
    core_end = evidence.end if core_end is None else core_end

    depth = evidence.depth
    ref_codes = evidence.ref_codes
    ref_count = evidence.ref_counts

    # Best alternate base: zero out the reference base and N, take the max.
    acgt = evidence.base_counts[:N].copy()
    columns = np.arange(acgt.shape[1])
    known_ref = ref_codes < N
    acgt[ref_codes[known_ref], columns[known_ref]] = 0
    best_alt = acgt.argmax(axis=0)
    snp_count = acgt[best_alt, columns]

    ins_count = evidence.counts[INS_AFTER]
    del_count = evidence.counts[DEL_AFTER]

    is_snp = _passes(snp_count, depth, config.min_alt_count, config.min_snp_fraction)
    is_ins = _passes(ins_count, depth, config.min_alt_count, config.min_indel_fraction)
    is_del = _passes(del_count, depth, config.min_alt_count, config.min_indel_fraction)

    in_core = np.zeros(len(depth), dtype=bool)
    in_core[core_start - evidence.start:core_end - evidence.start] = True
    keep = (is_snp | is_ins | is_del) & known_ref & (depth >= config.min_depth) & in_core
    idx = np.flatnonzero(keep)

    labels = [(is_snp, "SNP"), (is_ins, "INS"), (is_del, "DEL")]
    types = [",".join(name for mask, name in labels if mask[i]) for i in idx]

    return pd.DataFrame({
        "chrom": evidence.contig,
        "pos": evidence.start + idx + 1,
        "ref": [BASES[c] for c in ref_codes[idx]],
        "type": types,
        "depth": depth[idx],
        "ref_count": ref_count[idx],
        "alt_base": [BASES[c] for c in best_alt[idx]],
        "snp_count": snp_count[idx],
        "ins_count": ins_count[idx],
        "del_count": del_count[idx],
        "ins_len": modal_indel_lengths(evidence.indel_events, idx, 1),
        "del_len": modal_indel_lengths(evidence.indel_events, idx, 0),
    }, columns=CANDIDATE_COLUMNS)
