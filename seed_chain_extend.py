#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional


# =====================================================================
# Data Contracts & Signatures
# =====================================================================

@dataclass
class Anchor:
    """A seed match between read and reference."""
    read_pos: int   # 0-based coordinate on read
    ref_pos: int    # 0-based coordinate on reference contig
    k: int          # length of match (k-mer size)

@dataclass
class Chain:
    """Colinear sequence of anchors on a specific contig and strand."""
    contig: str
    strand: int     # 0 = '+', 1 = '-'
    anchors: List[Anchor]
    score: float

@dataclass
class AlignmentResult:
    """Standardized output consumable by any SAM/BAM writer."""
    contig: str
    strand: int
    ref_start: int
    ref_end: int
    read_start: int
    read_end: int
    cigar: str
    score: int
    mapq: int

BASE_MAP = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3,
    'a': 0, 'c': 1, 'g': 2, 't': 3
}
COMP_MAP = {0: 3, 1: 2, 2: 1, 3: 0}


def hash64(val: int) -> int:
    """Invertible 64-bit integer mix to distribute k-mer entropy uniformly."""
    val = (~val + (val << 21)) & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 24)
    val = (val + (val << 3) + (val << 8)) & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 14)
    val = (val + (val << 2) + (val << 4)) & 0xFFFFFFFFFFFFFFFF
    val = val ^ (val >> 28)
    val = (val + (val << 31)) & 0xFFFFFFFFFFFFFFFF
    return val


def extract_minimizers(seq: str, k: int = 15, w: int = 10) -> List[Tuple[int, int, int]]:
    """
    Extracts canonical (w, k)-minimizers from a nucleotide sequence.
    Handles 'N' (and non-ACGT) characters by segmenting the sequence
    into independent, contiguous valid sequence islands.
    """
    if len(seq) < k:
        return []

    mask = (1 << (2 * k)) - 1
    shift_rc = 2 * (k - 1)

    fwd_kmer = 0
    rev_kmer = 0
    valid_len = 0

    minimizers: List[Tuple[int, int, int]] = []
    current_island_kmers: List[Tuple[int, int, int]] = []

    def flush_island(kmers: List[Tuple[int, int, int]]) -> None:
        """Processes an isolated block of consecutive valid k-mers bounded by 'N' or sequence ends."""
        if not kmers:
            return

        n_kmers = len(kmers)

        # Case 1: The island is shorter than window w.
        # It cannot fill a window of size w, so select the single global minimum of the island.
        if n_kmers < w:
            min_kmer = min(kmers, key=lambda x: x[0])
            if not minimizers or minimizers[-1] != min_kmer:
                minimizers.append(min_kmer)
            return

        # Case 2: The island is at least w k-mers long.
        # Use the monotonic queue sliding window.
        window = []
        for i, entry in enumerate(kmers):
            while window and window[0] <= i - w:
                window.pop(0)
            while window and kmers[window[-1]][0] >= entry[0]:
                window.pop()
            window.append(i)

            if i >= w - 1:
                min_kmer = kmers[window[0]]
                if not minimizers or minimizers[-1] != min_kmer:
                    minimizers.append(min_kmer)

    # Stream through sequence bases
    for i, ch in enumerate(seq):
        if ch in BASE_MAP:
            b = BASE_MAP[ch]
            comp = COMP_MAP[b]
            fwd_kmer = ((fwd_kmer << 2) | b) & mask
            rev_kmer = (rev_kmer >> 2) | (comp << shift_rc)
            valid_len += 1

            if valid_len >= k:
                pos = i - k + 1
                if fwd_kmer <= rev_kmer:
                    current_island_kmers.append((hash64(fwd_kmer), pos, 0))
                else:
                    current_island_kmers.append((hash64(rev_kmer), pos, 1))
        else:
            # Hit 'N' or ambiguous IUPAC base:
            # 1. Process and flush the valid island prior to the gap
            if current_island_kmers:
                flush_island(current_island_kmers)
                current_island_kmers = []

            # 2. Reset rolling state
            fwd_kmer = 0
            rev_kmer = 0
            valid_len = 0

    # Flush any remaining valid island after the final base
    if current_island_kmers:
        flush_island(current_island_kmers)

    return minimizers