# src/deploy/state_dict_flex.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

# Common prefixes that often appear in saved checkpoints
COMMON_PREFIXES = ("head.", "model.", "module.")

def _normalize_keys_for_target(sd: Dict[str, torch.Tensor], target: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a copy of the state-dict with prefixes removed so its keys match
    the structure of `target`.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        kk = k

        # If loading into a plain Sequential head, strip the 'head.' prefix.
        if isinstance(target, nn.Sequential) and kk.startswith("head."):
            kk = kk[len("head."):]

        # Strip generic wrappers introduced by DDP/Lightning/HF, etc.
        for pref in ("model.", "module."):
            if kk.startswith(pref):
                kk = kk[len(pref):]

        out[kk] = v
    return out

def load_state_dict_flex(target: nn.Module, sd: Dict[str, torch.Tensor], strict: bool = True) -> None:
    """
    Load `sd` into `target`, automatically removing common prefixes.
    Raises with a helpful message if keys still don't match when `strict=True`.
    """
    norm = _normalize_keys_for_target(sd, target)
    result = target.load_state_dict(norm, strict=False)  # permissive first

    # Ignore benign BatchNorm tracking buffers when checking strictness
    missing = [k for k in result.missing_keys]
    unexpected = [k for k in result.unexpected_keys if not k.endswith("num_batches_tracked")]

    if strict and (missing or unexpected):
        msg = []
        if missing:
            msg.append(f"missing={missing}")
        if unexpected:
            msg.append(f"unexpected={unexpected}")
        raise RuntimeError("State dict mismatch after normalization: " + "; ".join(msg))
