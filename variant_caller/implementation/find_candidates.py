#!/usr/bin/env python3
"""Stage 1 of the variant caller: find candidates and build CNN examples.

    python find_candidates.py --bam reads.bam --ref ref.fasta --out-dir out/ \
        [--regions chr20 chr21:1-5000000] [--threads 8] \
        [--truth truth.vcf.gz [--bed confident.bed]]

Each 1 Mb tile is processed in one pass: count read evidence, find the
tile's candidates, then cut an example around each candidate from the same
in-memory evidence and save it as a shard. When all tiles are done the
shards are collated.

Writes to --out-dir:
    candidates.tsv   one row per candidate position (1-based pos) with evidence
    windows.bed      candidates <max_gap bp apart merged into windows (0-based)
    examples.npy     float16 (n_examples, n_features, width), one per candidate
    examples.tsv     metadata row i describes examples.npy[i] (pos, window, label)
    examples.json    feature names, width and shape
    config.json      thresholds used for this run

With --truth, examples are labelled with the number of alternate alleles at
that position (0 = reference). With --bed as well, only candidates inside the
confident regions get examples, since truth is undefined elsewhere.
"""

import argparse
import dataclasses
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pysam

from candidates import (CandidateConfig, collect_tile_evidence,
                        ensure_sorted_and_indexed, find_candidates,
                        merge_into_windows)
from candidates.candidates import CANDIDATE_COLUMNS
from candidates.collate import collate_shards
from candidates.examples import FEATURE_NAMES, build_examples
from candidates.truth import in_intervals, load_bed, load_truth_sites
from candidates.windows import assign_windows


# ---------------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------------

def parse_region(text, contig_lengths):
    """'chr20' or 'chr20:1000-2000' (1-based, inclusive) -> (contig, start0, end0)."""
    if ":" not in text:
        return text, 0, contig_lengths[text]
    contig, span = text.rsplit(":", 1)
    start, end = (int(x.replace(",", "")) for x in span.split("-"))
    return contig, start - 1, min(end, contig_lengths[contig])


def make_tiles(regions, tile_size):
    """Split each (contig, start, end) region into tiles of at most tile_size."""
    for contig, start, end in regions:
        for tile_start in range(start, end, tile_size):
            yield contig, tile_start, min(tile_start + tile_size, end)


# ---------------------------------------------------------------------------
# Worker process: each opens its own file handles once
# ---------------------------------------------------------------------------

_w = {}  # per-process state: bam, fasta, config, truth vcf, confident bed, shard dir


def _init_worker(bam_path, ref_path, config, truth_path, bed_path, shard_dir):
    _w["bam"] = pysam.AlignmentFile(bam_path, "rb")
    _w["fasta"] = pysam.FastaFile(ref_path)
    _w["config"] = config
    _w["truth"] = pysam.VariantFile(truth_path) if truth_path else None
    _w["bed"] = load_bed(bed_path) if bed_path else None
    _w["shard_dir"] = shard_dir


def label_examples(meta, contig, start, end):
    """Add label / truth columns to the tile's example metadata."""
    sites = load_truth_sites(_w["truth"], _w["fasta"], contig, start, end)
    empty = {"label": 0, "truth_kind": "", "truth_ref": "", "truth_alt": ""}
    rows = [sites.get(p, empty) for p in meta["pos"]]
    for key in empty:
        meta[key] = [r[key] for r in rows]
    return meta


def _process_tile(tile):
    """One pass over a tile: evidence -> candidates -> examples (-> labels)."""
    contig, start, end = tile
    config = _w["config"]
    half = config.example_width // 2

    # Evidence is collected with half an example of margin on each side so
    # candidates at the tile edge still get full context.
    contig_len = _w["fasta"].get_reference_length(contig)
    ev_start, ev_end = max(0, start - half), min(contig_len, end + half)
    evidence = collect_tile_evidence(_w["bam"], _w["fasta"], contig, ev_start, ev_end, config)
    cands = find_candidates(evidence, config, start, end)

    meta = cands[["chrom", "pos", "ref", "type"]].copy()
    if _w["bed"] is not None:
        meta = meta[[in_intervals(_w["bed"], contig, p - 1) for p in meta["pos"]]]
    if _w["truth"] is not None:
        meta = label_examples(meta, contig, start, end)
    else:
        meta["label"] = -1

    shard = None
    if len(meta):
        shard = os.path.join(_w["shard_dir"], f"{contig}_{start}.npy")
        np.save(shard, build_examples(evidence, meta["pos"].to_numpy(), config.example_width))
    return cands, meta, shard, evidence.indel_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    defaults = CandidateConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bam", required=True)
    p.add_argument("--ref", required=True, help="FASTA (plain or bgzipped) with .fai")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--regions", nargs="*", help="contigs or contig:start-end; default = all contigs")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--truth", help="truth VCF: label examples (training mode)")
    p.add_argument("--bed", help="confident regions: only these candidates get examples")
    p.add_argument("--full-sort-check", action="store_true",
                   help="scan every read to verify sort order instead of a prefix")
    for field in ("min_mapq", "min_base_quality", "min_alt_count", "min_depth",
                  "max_gap", "tile_size", "example_width"):
        p.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    for field in ("min_snp_fraction", "min_indel_fraction"):
        p.add_argument("--" + field.replace("_", "-"), type=float, default=getattr(defaults, field))
    p.add_argument("--no-supplementary", action="store_true")
    p.add_argument("--no-left-align", action="store_true",
                   help="count indels where the aligner put them instead of left-aligning")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    config = CandidateConfig(
        min_mapq=args.min_mapq, min_base_quality=args.min_base_quality,
        keep_supplementary=not args.no_supplementary,
        left_align_indels=not args.no_left_align,
        min_alt_count=args.min_alt_count, min_snp_fraction=args.min_snp_fraction,
        min_indel_fraction=args.min_indel_fraction, min_depth=args.min_depth,
        max_gap=args.max_gap, example_width=args.example_width, tile_size=args.tile_size,
    )

    bam_path = ensure_sorted_and_indexed(args.bam, args.out_dir, args.threads, args.full_sort_check)

    with pysam.AlignmentFile(bam_path, "rb") as bam, pysam.FastaFile(args.ref) as fasta:
        in_both = set(bam.references) & set(fasta.references)
        lengths = {c: n for c, n in zip(bam.references, bam.lengths) if c in in_both}
    region_texts = args.regions or [c for c in lengths]
    regions = [parse_region(r, lengths) for r in region_texts]
    tiles = list(make_tiles(regions, config.tile_size))
    print(f"[find_candidates] {len(tiles)} tiles over {len(regions)} region(s), {args.threads} worker(s)")

    # --- one pass per tile -------------------------------------------------------
    t0 = time.time()
    cand_parts, meta_parts, shards = [], [], []
    n_indels = n_shifted = 0
    init = (bam_path, args.ref, config, args.truth, args.bed, shard_dir)
    with Pool(args.threads, initializer=_init_worker, initargs=init) as pool:
        for i, (cands, meta, shard, stats) in enumerate(pool.imap(_process_tile, tiles), 1):
            cand_parts.append(cands)
            meta_parts.append(meta)
            shards.append(shard)
            n_indels += stats["indels"]
            n_shifted += stats["shifted"]
            if i % 10 == 0 or i == len(tiles):
                print(f"  {i}/{len(tiles)} tiles, {time.time() - t0:.0f}s")

    # --- collate once every tile is done ------------------------------------------
    cands = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    windows = merge_into_windows(cands, config.max_gap)
    meta = pd.concat(meta_parts, ignore_index=True)
    meta["window_id"] = assign_windows(meta, windows)
    n_examples = collate_shards(shards, os.path.join(args.out_dir, "examples.npy"), config.example_width)
    assert n_examples == len(meta), (n_examples, len(meta))
    os.rmdir(shard_dir)

    cands.to_csv(os.path.join(args.out_dir, "candidates.tsv"), sep="\t", index=False)
    windows.to_csv(os.path.join(args.out_dir, "windows.bed"), sep="\t", index=False, header=False)
    meta.to_csv(os.path.join(args.out_dir, "examples.tsv"), sep="\t", index=False)
    with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
        json.dump(dataclasses.asdict(config), fh, indent=2)
    with open(os.path.join(args.out_dir, "examples.json"), "w") as fh:
        json.dump({"shape": [n_examples, len(FEATURE_NAMES), config.example_width],
                   "dtype": "float16", "center_column": config.example_width // 2,
                   "features": FEATURE_NAMES, "labelled": bool(args.truth)}, fh, indent=2)

    print(f"[find_candidates] read indels counted: {n_indels}, left-shifted: {n_shifted} "
          f"({n_shifted / max(n_indels, 1):.1%})")
    print(f"[find_candidates] {len(cands)} candidates in {len(windows)} windows, "
          f"{n_examples} examples ({time.time() - t0:.0f}s) -> {args.out_dir}")
    if args.truth:
        print("[find_candidates] example labels:", meta["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
