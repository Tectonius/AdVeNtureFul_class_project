"""Aggregate per-position read evidence over one reference tile.

For a tile [start, end) we build, for every reference position:

  counts[channel, offset]   how many reads show each kind of observation
      A, C, G, T, N   aligned base seen in a read (ref-matching or not)
      DEL_SPAN        reference base is deleted in the read
      INS_AFTER       an insertion starts right after this base
      DEL_AFTER       a deletion starts right after this base
  rev_counts[...]           the same, from reverse-strand reads only
  sums[channel, offset]     running totals used for per-position means
      BQ_SUM          base qualities of the counted bases
      MAPQ_SUM        mapping qualities of the reads covering the base
                      (divide by mapq_depth, which also counts low-BQ bases)
      INS_LEN_SUM     lengths of the insertions anchored here
      DEL_LEN_SUM     lengths of the deletions anchored here

It also keeps a list of every read indel (anchor, kind, length), so that
candidates can report the most common indel length.

Indels are anchored on the preceding reference base, which is the same
convention VCF uses for indel POS, so candidate positions line up with truth.

Left-alignment: inside a repeat (e.g. deleting one A from AAAA) several
placements of an indel give the identical read, and aligners pick them
inconsistently. Before counting INS_AFTER / DEL_AFTER we slide each indel to
its leftmost equivalent position, so all reads supporting the same event
vote on the same base (and match left-normalized truth VCFs). Only the
anchor is moved; A/C/G/T/DEL_SPAN still follow the aligner's CIGAR, so depth
is unaffected.

Speed: human 48x coverage is ~150 billion aligned bases, far too many for a
per-base Python loop. We therefore loop over CIGAR *operations* only and use
numpy for every base inside an operation. Observations are recorded as flat
indices (channel * tile_len + offset) and tallied with np.bincount.
"""

import numpy as np
import pysam

# Channel rows of the count matrix.
A, C, G, T, N, DEL_SPAN, INS_AFTER, DEL_AFTER = range(8)
N_CHANNELS = 8
CHANNEL_NAMES = ["A", "C", "G", "T", "N", "DEL_SPAN", "INS_AFTER", "DEL_AFTER"]
BASES = "ACGTN"

# Rows of the sums matrix.
BQ_SUM, MAPQ_SUM, INS_LEN_SUM, DEL_LEN_SUM = range(4)
N_SUMS = 4

# ASCII byte -> base channel (anything unexpected counts as N).
BASE_CODE = np.full(256, N, dtype=np.int64)
for _code, _chars in enumerate(["Aa", "Cc", "Gg", "Tt"]):
    for _ch in _chars:
        BASE_CODE[ord(_ch)] = _code

# CIGAR operation codes (SAM spec).
CIGAR_MATCH, CIGAR_INS, CIGAR_DEL, CIGAR_SKIP, CIGAR_SOFT = 0, 1, 2, 3, 4
CIGAR_EQUAL, CIGAR_DIFF = 7, 8
ALIGNED_OPS = (CIGAR_MATCH, CIGAR_EQUAL, CIGAR_DIFF)


def encode_sequence(seq):
    """String of bases -> numpy array of channel codes (A=0 ... N=4)."""
    return BASE_CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


class ReferenceWindow:
    """Upper-cased reference bases for one contig, loaded around a tile.

    Reads overhang the tile edges, so indel shifting may need bases outside
    [start, end); the window grows on demand when that happens.
    """

    def __init__(self, fasta, contig, start, end, margin=100_000):
        self.fasta = fasta
        self.contig = contig
        self.length = fasta.get_reference_length(contig)
        self.margin = margin
        self._load(start - margin, end + margin)

    def _load(self, lo, hi):
        self.lo, self.hi = max(0, lo), min(self.length, hi)
        self.seq = self.fasta.fetch(self.contig, self.lo, self.hi).upper()

    def base(self, pos):
        if not self.lo <= pos < self.hi:
            self._load(min(self.lo, pos - self.margin), max(self.hi, pos + self.margin))
        return self.seq[pos - self.lo]

    def slice(self, start, end):
        return self.seq[start - self.lo:end - self.lo]


def left_align_deletion(ref, anchor, length, max_shift):
    """Leftmost anchor for a deletion of ref[anchor+1 : anchor+1+length].

    One step left is equivalent when the base before the deletion equals the
    last deleted base (ref[anchor] == ref[anchor + length]). E.g. deleting
    one A from C|AAAA: the anchor slides back to the C. `max_shift` stops us
    at the start of the aligned block before the indel.
    """
    shift = 0
    while shift < max_shift:
        p = anchor - shift
        if p < 0 or ref.base(p) == "N" or ref.base(p) != ref.base(p + length):
            break
        shift += 1
    return anchor - shift


def left_align_insertion(read_seq, q_ins, length, anchor, max_shift):
    """Leftmost anchor for an insertion of read_seq[q_ins : q_ins+length].

    One step left is equivalent when the read base before the insertion
    equals the last inserted base. Comparing read bases (not reference ones)
    keeps any mismatch in the read at the same reference position.
    """
    shift = 0
    while shift < max_shift:
        before = q_ins - 1 - shift
        if before < 0 or read_seq[before] != read_seq[before + length]:
            break
        shift += 1
    return anchor - shift


class TileEvidence:
    """All per-position evidence for reference positions [start, end) of one contig."""

    def __init__(self, contig, start, end, counts, rev_counts, sums,
                 indel_events, ref_codes, indel_stats, mapq_depth):
        self.contig = contig
        self.start = start
        self.end = end
        self.counts = counts            # (N_CHANNELS, L) both strands
        self.rev_counts = rev_counts    # (N_CHANNELS, L) reverse strand only
        self.sums = sums                # (N_SUMS, L) float
        self.indel_events = indel_events  # (n, 3) int: offset, is_insertion, length
        self.ref_codes = ref_codes      # reference base code per position
        self.indel_stats = indel_stats  # {"indels": n, "shifted": n}
        self.mapq_depth = mapq_depth    # reads counted in MAPQ_SUM (incl. low-BQ bases)

    @property
    def base_counts(self):
        return self.counts[A:N + 1]

    @property
    def depth(self):
        """Reads that cover each position, either with a base or a deletion."""
        return self.counts[A:N + 1].sum(axis=0) + self.counts[DEL_SPAN]

    @property
    def ref_counts(self):
        """Reads agreeing with the reference base (0 where the ref is N)."""
        return _pick_ref(self.counts, self.ref_codes)

    @property
    def rev_ref_counts(self):
        return _pick_ref(self.rev_counts, self.ref_codes)


def _pick_ref(counts, ref_codes):
    idx = np.minimum(ref_codes, N)
    return np.where(ref_codes < N, counts[idx, np.arange(counts.shape[1])], 0)


class _Tally:
    """Collects flat observation indices (optionally weighted) and bincounts them in batches."""

    def __init__(self, n_rows, tile_len, weighted=False, flush_every=5_000_000):
        self.shape = (n_rows, tile_len)
        self.size = n_rows * tile_len
        self.weighted = weighted
        self.totals = np.zeros(self.size, dtype=np.float64 if weighted else np.int64)
        self.chunks, self.weights = [], []
        self.pending = 0
        self.flush_every = flush_every

    def add(self, flat_indices, weights=None):
        if len(flat_indices):
            self.chunks.append(flat_indices)
            if self.weighted:
                self.weights.append(np.broadcast_to(weights, flat_indices.shape).astype(np.float64))
            self.pending += len(flat_indices)
            if self.pending >= self.flush_every:
                self.flush()

    def flush(self):
        if self.chunks:
            weights = np.concatenate(self.weights) if self.weighted else None
            self.totals += np.bincount(np.concatenate(self.chunks), weights=weights,
                                       minlength=self.size)
            self.chunks, self.weights, self.pending = [], [], 0

    def result(self):
        self.flush()
        return self.totals.reshape(self.shape)


class _TileAccumulator:
    """Everything `add_read` writes into while walking the reads of one tile."""

    def __init__(self, tile_len):
        self.tile_len = tile_len
        self.counts = _Tally(2 * N_CHANNELS, tile_len)  # rows 0-7 forward, 8-15 reverse
        self.sums = _Tally(N_SUMS, tile_len, weighted=True)
        self.indel_events = []                          # (offset, is_insertion, length)
        self.segments = []                              # (lo, hi, mapq) covered ref spans
        self.stats = {"indels": 0, "shifted": 0}


def _segment_sums(segments, tile_len):
    """Per-position (sum of MAPQ, number of reads) over covered [lo, hi) spans.

    Uses difference arrays: +value at lo, -value at hi, then a running sum.
    That costs two entries per alignment block instead of one per base.
    """
    if not segments:
        return np.zeros(tile_len), np.zeros(tile_len)
    lo, hi, mapq = np.asarray(segments, dtype=np.int64).T
    size = tile_len + 1
    mapq_diff = np.bincount(lo, mapq, size) - np.bincount(hi, mapq, size)
    read_diff = np.bincount(lo, minlength=size) - np.bincount(hi, minlength=size)
    return np.cumsum(mapq_diff)[:tile_len], np.cumsum(read_diff)[:tile_len]


def read_passes_filters(read, config):
    if read.is_unmapped or read.is_secondary or read.is_qcfail or read.is_duplicate:
        return False
    if read.is_supplementary and not config.keep_supplementary:
        return False
    return read.mapping_quality >= config.min_mapq


def add_read(read, tile_start, tile_end, ref, acc, config):
    """Record every observation `read` makes inside [tile_start, tile_end)."""
    L = acc.tile_len
    strand = N_CHANNELS if read.is_reverse else 0   # row offset into the counts tally
    mapq = read.mapping_quality
    read_seq = read.query_sequence
    seq = encode_sequence(read_seq)
    quals = read.query_qualities
    quals = np.asarray(quals) if quals is not None else np.full(len(seq), 255)

    def clip(lo, hi):
        return max(lo, tile_start), min(hi, tile_end)

    def record_indel(channel, is_insertion, length, original, shifted):
        # Anchors only move left, so indels that start left of the tile, or
        # that could never slide back into it, are someone else's.
        if tile_start <= shifted < tile_end:
            offset = shifted - tile_start
            indel_anchors.append((channel + strand) * L + offset)
            acc.indel_events.append((offset, is_insertion, length))
            acc.stats["indels"] += 1
            acc.stats["shifted"] += shifted != original

    indel_anchors = []  # flat indices for INS_AFTER / DEL_AFTER
    ref_pos = read.reference_start
    q_pos = 0
    aligned_run = 0     # aligned bases directly before the current position

    for op, length in read.cigartuples:
        if op in ALIGNED_OPS:
            lo, hi = clip(ref_pos, ref_pos + length)
            if lo < hi:
                q_lo = q_pos + (lo - ref_pos)
                bases = seq[q_lo:q_lo + (hi - lo)]
                bq = quals[q_lo:q_lo + (hi - lo)]
                good = bq >= config.min_base_quality
                offsets = np.arange(lo - tile_start, hi - tile_start)[good]
                acc.counts.add((bases[good] + strand) * L + offsets)
                acc.sums.add(BQ_SUM * L + offsets, bq[good])
                acc.segments.append((lo - tile_start, hi - tile_start, mapq))
            ref_pos += length
            q_pos += length
            aligned_run += length

        elif op in (CIGAR_INS, CIGAR_DEL):
            anchor = ref_pos - 1
            max_shift = aligned_run if config.left_align_indels else 0
            if tile_start <= anchor and anchor - max_shift < tile_end:
                if op == CIGAR_INS:
                    shifted = left_align_insertion(read_seq, q_pos, length, anchor, max_shift)
                    record_indel(INS_AFTER, 1, length, anchor, shifted)
                else:
                    shifted = left_align_deletion(ref, anchor, length, max_shift)
                    record_indel(DEL_AFTER, 0, length, anchor, shifted)

            if op == CIGAR_INS:
                q_pos += length
            else:
                lo, hi = clip(ref_pos, ref_pos + length)
                if lo < hi:
                    offsets = np.arange(lo - tile_start, hi - tile_start)
                    acc.counts.add((DEL_SPAN + strand) * L + offsets)
                    acc.segments.append((lo - tile_start, hi - tile_start, mapq))
                ref_pos += length
            aligned_run = 0

        elif op == CIGAR_SKIP:
            ref_pos += length
            aligned_run = 0

        elif op == CIGAR_SOFT:
            q_pos += length
        # hard clips (5) and padding (6) consume nothing

    acc.counts.add(np.asarray(indel_anchors, dtype=np.int64))


def collect_tile_evidence(bam, fasta, contig, start, end, config):
    """Build the TileEvidence for [start, end) from open pysam handles."""
    acc = _TileAccumulator(end - start)
    ref = ReferenceWindow(fasta, contig, start, end)
    last_start = -1

    for read in bam.fetch(contig, start, end):
        # Cheap runtime guard on the sort-order assumption.
        if read.reference_start < last_start:
            raise ValueError(f"BAM is not coordinate-sorted near {contig}:{read.reference_start}")
        last_start = read.reference_start

        if read_passes_filters(read, config):
            add_read(read, start, end, ref, acc, config)

    by_strand = acc.counts.result()
    rev_counts = by_strand[N_CHANNELS:]
    events = np.asarray(acc.indel_events, dtype=np.int64).reshape(-1, 3)
    sums = acc.sums.result()
    offsets, is_ins, lengths = events.T
    L = end - start
    sums[INS_LEN_SUM] = np.bincount(offsets[is_ins == 1], lengths[is_ins == 1], minlength=L)
    sums[DEL_LEN_SUM] = np.bincount(offsets[is_ins == 0], lengths[is_ins == 0], minlength=L)
    sums[MAPQ_SUM], mapq_depth = _segment_sums(acc.segments, L)
    return TileEvidence(
        contig, start, end, mapq_depth=mapq_depth,
        counts=by_strand[:N_CHANNELS] + rev_counts,
        rev_counts=rev_counts,
        sums=sums,
        indel_events=events,
        ref_codes=encode_sequence(ref.slice(start, end)),
        indel_stats=acc.stats,
    )


def evidence_for_region(bam_path, ref_path, contig, start, end, config):
    """Convenience wrapper that opens the files itself (handy in notebooks)."""
    with pysam.AlignmentFile(bam_path, "rb") as bam, pysam.FastaFile(ref_path) as fasta:
        return collect_tile_evidence(bam, fasta, contig, start, end, config)
