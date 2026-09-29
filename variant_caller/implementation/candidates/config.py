"""All tunable thresholds for candidate generation, in one place.

Defaults follow DeepVariant's "very sensitive caller" where there is an
equivalent (min 2 supporting reads, 12% allele fraction for SNPs, 6% for
indels). Candidate generation should be *sensitive*: a false candidate only
costs one extra example for the CNN, a missed one can never be recovered.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateConfig:
    # --- read filters -------------------------------------------------------
    min_mapq: int = 5               # skip reads with mapping quality below this
    min_base_quality: int = 10      # aligned bases below this are not counted
    keep_supplementary: bool = True  # split-read pieces of long reads
    left_align_indels: bool = True  # slide indels to leftmost equivalent position

    # --- candidate thresholds ----------------------------------------------
    min_alt_count: int = 2          # reads that must support the alternate
    min_snp_fraction: float = 0.12  # alt reads / depth for a SNP candidate
    min_indel_fraction: float = 0.06  # indel reads / depth for an indel candidate
    min_depth: int = 1              # ignore positions with lower coverage

    # --- windowing -----------------------------------------------------------
    max_gap: int = 100              # candidates closer than this share a window

    # --- examples ------------------------------------------------------------
    example_width: int = 221        # reference positions per example, candidate centred

    # --- performance -----------------------------------------------------------
    tile_size: int = 1_000_000      # bases of reference processed per work unit
