"""A-MAC-style admission scoring (U/C/N/R/T) for LME candidate ingest."""

from .scorer import AmacAdmissionScorer, AmacCandidate, AmacFeatures, parse_amac_weights_arg

__all__ = [
    "AmacAdmissionScorer",
    "AmacCandidate",
    "AmacFeatures",
    "parse_amac_weights_arg",
]
