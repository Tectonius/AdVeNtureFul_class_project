"""Make sure the input BAM is coordinate-sorted and indexed.

Every later step assumes reads arrive ordered by reference start position and
that we can jump to any region with the index. This module checks that once,
up front, and fixes it (sort + index) when it is not the case.
"""

import os

import pysam


def header_says_sorted(bam_path):
    """True if the @HD line declares SO:coordinate."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return bam.header.to_dict().get("HD", {}).get("SO") == "coordinate"


def scan_is_sorted(bam_path, max_reads=None):
    """Walk the reads and confirm (contig, start) never decreases.

    The header can lie (e.g. after a careless `samtools cat`), so this does
    the real check. `max_reads` limits it to a prefix for a quick sanity test.
    """
    last_contig, last_start = -1, -1
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for n, read in enumerate(bam.fetch(until_eof=True)):
            if max_reads is not None and n >= max_reads:
                break
            if read.is_unmapped and read.reference_id < 0:
                break  # unplaced unmapped reads sit at the end of a sorted BAM
            key = (read.reference_id, read.reference_start)
            if key < (last_contig, last_start):
                return False
            last_contig, last_start = key
    return True


def has_index(bam_path):
    return any(os.path.exists(bam_path + ext) for ext in (".bai", ".csi")) or \
        os.path.exists(os.path.splitext(bam_path)[0] + ".bai")


def ensure_sorted_and_indexed(bam_path, work_dir, threads=4, full_scan=False):
    """Return the path of a coordinate-sorted, indexed version of `bam_path`.

    If the BAM already is one, it is returned unchanged. Otherwise a sorted
    copy is written into `work_dir`. `full_scan=True` verifies sort order by
    reading every record instead of trusting the header plus a prefix check.
    """
    sorted_ok = header_says_sorted(bam_path) and scan_is_sorted(
        bam_path, max_reads=None if full_scan else 100_000
    )

    if not sorted_ok:
        os.makedirs(work_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(bam_path))[0]
        sorted_path = os.path.join(work_dir, base + ".sorted.bam")
        print(f"[bam_prep] {bam_path} is not coordinate-sorted; sorting to {sorted_path}")
        pysam.sort("-@", str(threads), "-o", sorted_path, bam_path)
        bam_path = sorted_path

    if not has_index(bam_path):
        print(f"[bam_prep] indexing {bam_path}")
        pysam.index("-@", str(threads), bam_path)

    return bam_path
