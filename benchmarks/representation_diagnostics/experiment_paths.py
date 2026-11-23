"""Resolve historical OncoBERT paths without rewriting archived provenance.

Only paths rooted at the former experiment repository are remapped. External
GENIE, TCGA and GenePT input paths and checkpoint tensors remain unchanged.
"""
import json
from pathlib import Path
import os

ROOT = Path(os.environ.get(
    "ONCOBERT_EXPERIMENT_ROOT",
    Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments",
))
LEGACY_ROOTS = tuple(filter(None, os.environ.get(
    "ONCOBERT_LEGACY_ROOTS", ""
).split(":")))


def relocate_paths(value):
    if isinstance(value,str):
        for old in LEGACY_ROOTS:
            if value==old:return str(ROOT)
            if value.startswith(old+'/'):return str(ROOT)+value[len(old):]
        return value
    if isinstance(value,dict):return {relocate_paths(k):relocate_paths(v) for k,v in value.items()}
    if isinstance(value,list):return [relocate_paths(v) for v in value]
    if isinstance(value,tuple):return tuple(relocate_paths(v) for v in value)
    return value


def relocated_json_loads(*args,**kwargs):return relocate_paths(json.loads(*args,**kwargs))
def relocated_json_load(*args,**kwargs):return relocate_paths(json.load(*args,**kwargs))


def relocate_checkpoint(checkpoint):
    # Do not traverse/copy state dictionaries, tensors or optimizer buffers.
    if isinstance(checkpoint,dict) and 'config' in checkpoint:
        return {**checkpoint,'config':relocate_paths(checkpoint['config'])}
    return checkpoint
