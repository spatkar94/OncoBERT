"""Compatibility imports for the canonical model in :mod:`oncobert.model`."""

from .model import (
    CleanMaskedSetModel,
    OncoBERT,
    OncoBERTSetTransformer,
    STATE_NAMES,
    masked_ce,
    masked_cross_entropy,
    uniform_mask,
)

__all__ = [
    "OncoBERT",
    "OncoBERTSetTransformer",
    "CleanMaskedSetModel",
    "STATE_NAMES",
    "uniform_mask",
    "masked_cross_entropy",
    "masked_ce",
]
