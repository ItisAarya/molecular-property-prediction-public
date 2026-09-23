# v1.1.0 — review corrections

A full reproduction and review of v1.0.0. The archive reproduced exactly, but the review found
a label leak and several claims the archive did not support. **The v1.0.0 findings below are
superseded where they conflict with this section.**

## The notation leak

In ClinTox and BBBP the notation of a SMILES string (aromatic or Kekulé) predicts the label:
notation-bit AUC 1.000 for ClinTox toxicity and 0.827 for BBBP permeability. Only models that
read SMILES characters can use it. `src/data/smiles.py` now canonicalises both datasets wherever
a sequence model reads them, and `scripts/audit_notation.py` audits any dataset.

- Re-run on canonical strings: the CPU cached ladder (`fuse_concat`, `fuse_gated`, `fuse_xattn`,
  `fuse_bilinear`, `fuse_proposed`, `fuse_gated_nograph`), a sequence-only control
  (`fuse_seqonly`) and the deployed models (`deploy_proposed`). The raw-string originals are
  archived as `<tag>_rawsmiles`.
- Not re-run (GPU or pipeline runs): LoRA, the frozen ChemBERTa view, the T4 ladder, the rank
  sweep, the end-to-end ladder, and the pipeline's `trf`, `hybrid` and `ens`. Their ClinTox and
  BBBP results are excluded (`src/eval/leakage.py`), and their comparisons run over six
  datasets.
- Effect: frozen ChemBERTa fell from 0.988 to 0.795 AUC on ClinTox, and every fusion model fell
  by 0.041–0.150 AUC across the two datasets (`scripts/leakage_effect.py`).

## Findings that changed

- `proposed` vs the two-layer GIN: favoured on 7 of 8, p = 0.070 / 0.109 / 0.078 (was 8 of 8,
  p = 0.0078). Only FreeSolv survives correction.
- `proposed` vs `desc`: favoured on 3 of 8 (p ≥ 0.727). `gated` and `bilinear` are worse than
  `desc` on 7 of 8 (Wilcoxon p = 0.023).
- "A 16,513-parameter gate matches the 1,169,793-parameter block" was not supported:
  `proposed` is favoured over `gated` on 6 of 8 (Wilcoxon p = 0.039, sign p = 0.289), and
  equivalence holds on 2 of 8.
- Bilinear vs concatenation is no longer significant on the CPU ladder (6 of 8, p = 0.312).
  It is still significant on both T4 ladders (6 of 6, p = 0.031).
- Conformal minority coverage: a control (`desc_nopw`, trained without class weighting) shows
  the failure follows class weighting, not architecture (77.9% → 16.3% of actives at unchanged
  AUC). Set size is a symptom of the same cause, not an independent predictor.
- APS and RAPS: only the deterministic forms collapse at two classes; the randomised forms
  (`--randomized`) behave as intended.
- Device: CPU vs T4 moves 58 of 240 single-split results (24%) past the practical threshold.
  A seed change on the same CPU moves 25 of 96 (26%), so the device effect is ordinary
  seed variance.
- Duplicates: seeded splits do put up to two duplicate groups in different parts of a split (BBBP,
  ESOL). The canonical split does not.
- Split conventions: up to 0.223 AUC apart on the same model (113 pairs).

## Code and archive fixes

- `configs/shared.yaml` now matches the trainers (`representation.head_dropout: 0.2`,
  `fusion.xattn_dropout: 0.1`), and `scripts/check_configs.py` checks constructor defaults
  (35 checks). `src/deploy/predict.py` reads the renamed keys.
- `scripts/run_comparisons.py` regenerates all 59 paired comparisons over all eight datasets;
  stale subset comparisons were removed.
- New analyses: `scripts/equivalence.py` (TOST), `scripts/device_effect.py` (with a seed-43
  control), `scripts/calibration_summary.py`, `scripts/validate_conformal.py` (synthetic
  coverage check and temperature-scaling invariance, both quoted in section 6), and a duplicate
  audit across every split.
- `--no-pos-weight` in `src/train/train_view.py`; randomised APS/RAPS in `src/eval/conformal.py`.
- Paper: rewritten draft with generated tables, `scripts/check_paper.py` (18 tables and 62
  prose claims), a LaTeX builder with numbered floats and cross-references, 66 references with
  dataset and software citations, and corrected metadata for three entries.
- `constraints.txt` pins `rdkit==2025.3.5`, the version the archive was built with.

## Independent verification of this release

- **Data rebuilt from scratch.** From a clean export, the README pipeline (prep, graphs, tokens,
  embeddings, pool, descriptors, splits) produced a data pool and 48 split files bit-identical
  to those the archive was trained on, including the canonical-SMILES ClinTox and BBBP
  embeddings, and `verify_prep` passed.
- **CPU models re-trained from that rebuild** on the same Intel Core i7-9750H: the seven
  canonical-SMILES models behind Table 3 and the fusion results (`fuse_seqonly`,
  `fuse_gated_nograph`, `fuse_gated`, `fuse_proposed`, `fuse_concat`, `fuse_xattn`,
  `fuse_bilinear` on ClinTox and BBBP), the class-weighting control `desc_nopw` and the seed-43
  control. All 348 metric files and all 180 locally archived prediction arrays matched the
  archive bit for bit. The motif probe (Table 14) re-ran to identical output.
- **Statistics recomputed with independent code** (no project imports) from the per-split
  metrics and raw predictions: every row of Tables 8, 9 and 11-13; Tables 2-6, 15, 16, 18 and
  20; the absolute-residual and normalised rows of Table 17; the split-gap numbers, gate weights
  and equivalence counts. Table 19 was re-aggregated from the archived per-split cliff results.
  No mismatches.
- **References.** All 55 entries with a DOI or arXiv identifier were checked against Crossref
  and arXiv (title, first author, year).
- **Fixed in this pass:** stale `paper/tables/`; a synthetic-validation number that no script
  produced and a "probabilities moved by up to 0.02" claim the predictions contradict (up to
  0.26), both now produced by `scripts/validate_conformal.py`; wording in sections 5.4 and 5.6;
  the README citation title; and CITATION.cff / .zenodo.json abstracts that still stated
  superseded v1.0.0 findings.
- **Not re-trained** (GPU runs on Colab): LoRA, the frozen ChemBERTa view, AttentiveFP,
  Chemprop, the T4 ladder, the rank sweep and the end-to-end ladder. Their archived per-split
  results were used as they are; only the statistics built on them were recomputed.

## Known limitations of this release

- The GPU and pipeline models listed above were not re-run on canonical SMILES, so their
  ClinTox and BBBP results stay excluded and their comparisons run over six datasets.
- Per-split predictions are not in the repository. The CPU-trained models regenerate them by
  re-training; predictions for the GPU-trained models are available from the author on request.

---

# v1.0.0 — Phases 0–5 complete

An evaluation protocol for multi-view molecular property prediction, a fusion architecture
measured under it, and the results — most of which are negative.

## What is in it

**The protocol.** Eight MoleculeNet datasets, six scaffold splits (the DeepChem canonical one
plus five seeded), 80/10/10, paired per-dataset comparisons with Holm correction, an
across-dataset test reported on three statistics, and a stated minimum detectable effect
(±0.02 AUC, ±0.10 RMSE). One hyper-parameter setting held identical across every model,
machine-checked against the training code.

**The models.** Five view encoders (GIN, GINE, AttentiveFP, ChemBERTa frozen and with LoRA
adapters, ECFP + 217 RDKit descriptors), a five-rung fusion ladder (`concat`, `gated`,
`xattn`, `bilinear`, `proposed`) measured in two training settings, two published external
baselines (AttentiveFP, Chemprop), and the inherited pipeline preserved runnable. 1,254
archived (model, dataset, split) fits in total.

**Uncertainty.** Split-conformal prediction across fifteen models — LAC, APS, RAPS,
class-conditional (Mondrian), CQR, normalised and absolute residual scores — plus post-hoc
calibration over 65 (model, dataset) pairs.

**A live demo.** `python -m streamlit run app/predict_app.py` predicts all eight properties from a typed
SMILES string, with a conformal prediction set or interval, the model's measured test score,
and an honest label saying whether the molecule was in that dataset's training data.

## Findings

- The architecture beats the **baseline graph encoder** on **8 of 8 datasets** (p = 0.0078 on
  three across-dataset statistics) — and **does not** beat a fingerprint-plus-descriptor MLP
  (4 of 8, p = 0.55). Against the earlier pipeline's *best* configuration, the stacking
  meta-learner, it improves on **1 of 5** datasets after correction with no across-dataset
  difference.
- A **16,513-parameter** gate matches the **1,169,793-parameter** fusion block.
- Roughly **eleven GPU-hours** of LoRA adaptation buy nothing over a cached frozen embedding.
- Neither published external baseline is distinguishable from the inherited two-layer GIN.
- A nominal **90% conformal guarantee covers 9.7–14.1% of active compounds** for three of
  fifteen models, Chemprop among them; mean prediction-set size predicts which models fail,
  with a 57-point gap and nothing in between.
- Temperature scaling **provably cannot** alter a binary conformal set; APS and RAPS are
  unusable at two classes.
- A fully seeded pipeline moved to a **different GPU is a different model** — 18% of
  single-split numbers move by more than the effect size the field reports differences at,
  while the five-split mean absorbs it entirely.
- **The significance of our own mechanism claim is not stable** to the bilinear rank, a
  constant chosen once. The effect's direction holds at every rank; its p-value does not.
- Three of the eight benchmarks contain the **same molecule twice with conflicting labels**
  (ClinTox: 19 groups in 1,480 molecules). Not leakage — no duplicate group crosses a split,
  because the scaffold split keeps identical molecules together — but a ceiling on accuracy.

Three of the plan's four success criteria come back negative or qualified. The draft says so.

## Bugs found in the inherited pipeline, all of which changed numbers

A missing-label mask that fabricated 24% of Tox21's canonical test labels as negatives;
model selection performed on the test split; a 0.22 AUC gap between two procedures both called
"scaffold split"; a significance rule that tested against the wrong quantity; and a head-width
confound.

## Verifying it

```bash
python -m scripts.check_configs   # 27 assertions: the config matches the trainers
python -m scripts.check_paper     # 185 assertions: the draft matches the archives
python -m scripts.check_deploy    # the live featuriser matches the training features
```

All three pass on this tag.

## Not included

Hyper-parameter tuning (deliberately — see `src/tune/study.py`) and two of the three
leave-one-view-out arms.

The motif view was not built either, but it is no longer merely absent:
`python -m scripts.probe_motifs` measures what BRICS fragments and Murcko scaffolds add to
the views that are in the ladder, and §5.7 of the draft reports the answer. Over the five
seeded splits, fragments lose to an ECFP-plus-descriptor baseline on 8 of 8 datasets and add
nothing that clears the minimum detectable effect; the canonical split disagrees, and both are
reported. Across all 48 dataset-split pairs no ring-bearing test molecule shares a Murcko
scaffold with any training molecule — which is what a scaffold split is for.
