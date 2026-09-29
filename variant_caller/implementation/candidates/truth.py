"""Truth-set helpers: VCF normalization, confident-region BEDs, labels.

Used both to label training examples and by evaluate_candidates.py, so the
two always agree on what "the truth variant at this position" means.
"""

import bisect
from collections import defaultdict


def normalize(fasta, chrom, pos, ref, alt):
    """Left-align and trim one VCF allele pair (the `bcftools norm -f` rule).

    Truth VCFs are not always left-normalized, while our candidates always
    sit on the leftmost equivalent anchor. Returns (pos, ref, alt), 1-based.
    """
    ref, alt = ref.upper(), alt.upper()
    while True:
        if ref and alt and ref[-1] == alt[-1]:
            ref, alt = ref[:-1], alt[:-1]        # drop shared last base
        elif (not ref or not alt) and pos > 1:
            base = fasta.fetch(chrom, pos - 2, pos - 1).upper()
            ref, alt, pos = base + ref, base + alt, pos - 1  # extend one base left
        else:
            break
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1  # drop shared first base
    return pos, ref, alt


def truth_kind(ref, alt):
    if len(ref) == len(alt) == 1:
        return "SNP"
    if len(ref) == 1:
        return "INS"
    if len(alt) == 1:
        return "DEL"
    return "COMPLEX"


def load_bed(path):
    """{contig: sorted list of (start, end)} for 0-based half-open intervals."""
    intervals = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith(("#", "track")):
                c, s, e = line.split()[:3]
                intervals[c].append((int(s), int(e)))
    return {c: sorted(v) for c, v in intervals.items()}


def in_intervals(intervals, contig, pos0):
    ivs = intervals.get(contig, [])
    i = bisect.bisect_right(ivs, (pos0, float("inf"))) - 1
    return i >= 0 and ivs[i][0] <= pos0 < ivs[i][1]


def alt_allele_count(record):
    """Number of non-reference alleles in the first sample's genotype.

    Haploid "1" -> 1; diploid "0/1" -> 1, "1/1" or "1/2" -> 2. Records with no
    genotype are treated as a single alternate allele.
    """
    if not record.samples:
        return 1
    gt = record.samples[0].get("GT") or (1,)
    return sum(1 for a in gt if a not in (None, 0)) or 1


def load_truth_sites(vcf, fasta, contig, start, end, pad=1000):
    """Normalized truth variants with POS in [start, end) (0-based bounds).

    Returns {pos (1-based): {"label", "truth_kind", "truth_ref", "truth_alt"}}.
    Several records on one POS (e.g. a SNP and an indel sharing an anchor)
    are merged: the label is the largest alt-allele count, the rest joined.
    """
    if contig not in vcf.header.contigs:
        return {}
    sites = {}
    # Records a little to the right can normalize leftwards into the range.
    for rec in vcf.fetch(contig, max(0, start - 1), end + pad):
        gt = rec.samples[0].get("GT") if rec.samples else None
        called = [a for a in (gt or (1,)) if a not in (None, 0)]
        alt = rec.alleles[called[0]] if called and called[0] < len(rec.alleles) else rec.alts[0]
        pos, ref_n, alt_n = normalize(fasta, contig, rec.pos, rec.ref, alt)
        if not start < pos <= end:
            continue
        site = sites.setdefault(pos, {"label": 0, "truth_kind": [], "truth_ref": [], "truth_alt": []})
        site["label"] = max(site["label"], alt_allele_count(rec))
        site["truth_kind"].append(truth_kind(ref_n, alt_n))
        site["truth_ref"].append(ref_n)
        site["truth_alt"].append(alt_n)
    for site in sites.values():
        for key in ("truth_kind", "truth_ref", "truth_alt"):
            site[key] = ",".join(site[key])
    return sites
