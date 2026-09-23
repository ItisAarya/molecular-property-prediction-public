"""
src/eval/leakage.py

Which archived results must not be compared, because their inputs leaked the label.

    from src.eval.leakage import excluded
    if excluded("clintox", "lora"): ...

WHY
---
On ClinTox and BBBP the way a SMILES string is written predicts the label (see
`src/data/smiles.py` and `scripts/audit_notation.py`). Every model that reads raw SMILES
characters -- the ChemBERTa views, every fusion model that includes the sequence view, and the
inherited pipeline's transformer and the two meta-models built on it -- was originally trained
on those raw strings.

The CPU-trained cached fusion ladder was re-run on canonical strings for those two datasets, and
the canonical runs replaced the archived ones under the same tags. The raw-string runs are kept
under `<tag>_rawsmiles`, because the size of the leak is itself a result (Section 4.1 and
Table 3 of the paper). The deployed models (`deploy_proposed`) were re-trained the same way.

The GPU-only runs -- LoRA, the frozen ChemBERTa view as trained on a T4, the T4 cached ladder,
the rank sweep and the end-to-end ladder -- were not re-run. Their ClinTox and BBBP results are
excluded here, so every comparison involving them is over the six datasets whose notation is
uniform, and says so. The inherited pipeline's `trf`, `hybrid` and `ens` are excluded on the same
two datasets for the same reason.
"""

from src.data.smiles import NOTATION_CANONICALISED

# Raw-SMILES runs on ClinTox/BBBP that were NOT re-run on canonical strings.
RAW_SMILES_EXCLUDED = (
    "seq_frozen", "lora",
    "trf", "hybrid", "ens",
    "fuse_concat_gpu", "fuse_xattn_gpu", "fuse_bilinear_gpu", "fuse_proposed_gpu",
    "fuse_bilinear_r16_gpu", "fuse_bilinear_r32_gpu", "fuse_bilinear_r128_gpu",
    "fuse_concat_e2e", "fuse_gated_e2e", "fuse_xattn_e2e", "fuse_bilinear_e2e",
    "fuse_proposed_e2e",
)

# Re-run on canonical strings; their raw-string originals are archived as `<tag>_rawsmiles`.
RERUN_CANONICAL = ("fuse_concat", "fuse_gated", "fuse_xattn", "fuse_bilinear", "fuse_proposed",
                   "fuse_gated_nograph", "deploy_proposed")


def excluded(dataset, tag):
    """True if this (dataset, model) result read raw SMILES on a notation-leaking dataset."""
    return dataset in NOTATION_CANONICALISED and tag in RAW_SMILES_EXCLUDED
