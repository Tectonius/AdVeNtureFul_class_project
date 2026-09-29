"""Heuristic candidate generation: BAM -> candidate positions -> windows."""

from .bam_prep import ensure_sorted_and_indexed
from .candidates import find_candidates
from .config import CandidateConfig
from .evidence import TileEvidence, collect_tile_evidence, evidence_for_region
from .windows import merge_into_windows
