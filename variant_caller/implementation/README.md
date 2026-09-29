# Variant caller implementation

## Stage 1 — candidates + CNN examples (`find_candidates.py`)

One pass per 1 Mb tile, then a collation step:

```
BAM ──bam_prep──> sorted+indexed BAM
per tile (parallel):
    ──evidence───> count / strand / quality matrices over the tile (+110 bp margin)
    ──candidates─> candidate rows for the tile core
    ──examples───> (n, 21 features, 221 bp) centred on each candidate -> shard .npy
    ──truth──────> label = alt allele count at that POS (with --truth)
after all tiles:
    ──windows────> windows.bed; each example tagged with its window_id
    ──collate────> examples.npy + examples.tsv (row i describes example i)
```

| module | job |
|---|---|
| `candidates/config.py` | every threshold, one dataclass |
| `candidates/bam_prep.py` | check header `SO:coordinate` + scan read order; sort/index if needed |
| `candidates/evidence.py` | walk each read's CIGAR, count A/C/G/T/N, deletion-span, insertion-after, deletion-after per reference base |
| `candidates/candidates.py` | apply SNP / indel count + fraction thresholds, keep ref-support counts |
| `candidates/examples.py` | per-position features -> fixed-width examples; `load_examples()` |
| `candidates/truth.py` | VCF left-normalization, confident BED, labels |
| `candidates/windows.py` | chain nearby candidates into BED windows, map examples to windows |
| `candidates/collate.py` | join per-tile shards into one memory-mapped `examples.npy` |

The genome is processed in 1 Mb tiles (index `fetch`) so memory stays flat
and tiles run in parallel. Indels are anchored on the preceding reference
base, which matches VCF POS.

```bash
python find_candidates.py --bam reads.bam --ref ref.fasta --out-dir out/ --threads 8 \
    [--regions chr20] [--truth truth.vcf.gz --bed confident.bed]   # labels = training mode
python evaluate_candidates.py --out-dir out/ --truth truth.vcf.gz --ref ref.fasta \
    [--bam reads.bam --threads 8] [--bed confident.bed]
```

Indels are slid to their leftmost equivalent position before counting
(`--no-left-align` turns this off). The evaluator left-normalizes the truth
VCF the same way, and with `--bam` writes `out/missed.tsv` giving a reason
for every truth variant without a candidate at its POS.

Coordinates: `candidates.tsv` `pos` is 1-based; `windows.bed` is 0-based half-open.

Examples: `X, meta = candidates.examples.load_examples("out/")` gives a
memory-mapped float16 array `(n, 21, 221)` and its metadata table. The
candidate is column 110; feature names are in `out/examples.json` and are
documented at the top of `candidates/examples.py`.
