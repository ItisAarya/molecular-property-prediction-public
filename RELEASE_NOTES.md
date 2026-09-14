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
python -m scripts.check_paper     # 165 assertions: the draft matches the archives
python -m scripts.check_deploy    # the live featuriser matches the training features
```

All three pass on this tag.

## Not included

Hyper-parameter tuning (deliberately — see `src/tune/study.py`) and two of the three
leave-one-view-out arms.

The motif view was not built either, but it is no longer merely absent:
`python -m scripts.probe_motifs` measures what BRICS fragments and Murcko scaffolds add to
the views that are in the ladder, and §5.7 of the draft reports the answer. Fragments lose to
an ECFP-plus-descriptor baseline on 8 of 8 datasets and add nothing measurable on top of it,
and across all 48 dataset-split pairs no test molecule shares a Murcko scaffold with any
training molecule — which is what a scaffold split is for.
