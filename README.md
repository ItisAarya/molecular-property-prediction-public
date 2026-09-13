# Multi-view molecular property prediction, evaluated honestly

Eight MoleculeNet datasets, five molecular representations, five ways of fusing them,
and distribution-free uncertainty on top — all measured under one protocol, across six
scaffold splits, with the negative results reported.

The headline is not an architecture. It is a protocol strict enough to separate a real gain
from a favourable split — and four results it returned: a **71x smaller fusion block that
performs identically**, a **free diagnostic** for which models abandon their minority class,
a proof that one standard calibration step cannot help, and a reproducibility requirement
that costs one extra column.

---

## What this is

An earlier MoleculeNet pipeline of our own, rebuilt to answer one question properly: *does
combining molecular representations actually help?* Answering it required first fixing an
evaluation that could not have detected the answer either way.

**What the evaluation found in that earlier code** (all real, all changed numbers):

- Tox21's missing-label mask was dropped, so **24% of canonical test labels were fabricated
  as negatives**.
- Regression metrics were reported in z-scored units, not logS / logD / kcal·mol⁻¹.
- Model selection ran on the **test** split.
- The meta-learner, ensemble weights, calibrators and decision thresholds were all fitted
  on a single validation split.
- The "frozen" ChemBERTa ran with dropout active; computed `edge_attr` was discarded.
- Nothing was seeded.

**What the fixed evaluation then found about the new work:**

Counts below are datasets where the candidate wins, out of 8, **after** Holm–Bonferroni
correction; the uncorrected count is given where it differs, because that is the number an
uncorrected analysis would have reported.

| Question | Answer |
|---|---|
| Does an edge-aware graph encoder beat the baseline 2-layer GIN? | No — **0** of 8 (1 uncorrected), at 5× the parameters |
| Does LoRA fine-tuning beat a frozen transformer? | Only on **ESOL** — 1 of 8 (2 uncorrected), and both raw wins were regression, where the frozen encoder was weak |
| Does concatenating views beat the best single view? | No — 0 of 8 |
| Does the proposed cross-attention + bilinear fusion beat a 16.5k-parameter gate? | No — **0** of 8 (1 uncorrected: BACE at p=0.017 → 0.135), for 71× the parameters |
| Does end-to-end adaptation beat cached frozen embeddings? | No — 0 of 8, for ~11 GPU-hours |
| Is the graph view needed at all? | No — 0 of 8 either way; dropping it runs 13× faster on 59% of the parameters |
| Does the fusion model beat the baseline GIN *across datasets*? | **Yes** — favoured on 8/8, p=0.0078 on all three across-dataset statistics |
| Does it beat the best single view (`desc`) across datasets? | No — 4/8, p=0.46 |
| Does a 90% conformal guarantee cover 90% of actives? | No — 72–78%, and 9.8% in the worst case |

The last three rows are the point. The fusion model **is** better than the baseline it
started from, and that survives every test. It is **not** better than a descriptor MLP, and no
amount of architecture in between changed that.

---

## Quick start

```bash
conda env create -f environment.yml && conda activate mpp
python -m scripts.prep_moleculenet          # DeepChem -> CSV + npz
python -m scripts.make_graphs               # RDKit graphs (34 atom / 7 bond dims)
python -m scripts.make_descriptors          # 217 RDKit 2-D descriptors, raw
python -m scripts.tokenize_smiles           # ChemBERTa token ids
python -m scripts.cache_embeddings          # frozen-encoder embeddings
python -m scripts.verify_prep               # assertions over every artifact
python -m scripts.build_pool                # per-split artifacts -> data/pool/
python -m scripts.make_splits               # deepchem + seeds 0-4, audits leakage
```

Then train and compare:

```bash
python -m scripts.run_view_multiseed --tags desc --variants deepchem seed0 seed1 seed2 seed3 seed4
python -m src.eval.view_stats --a fuse_gated --b desc --datasets tox21 bbbp clintox esol lipophilicity bace sider freesolv
```

`python -m scripts.check_configs` asserts that `configs/shared.yaml` still describes the
trainers' actual defaults. Run it before trusting a comparison.

### Predicting a molecule you type in

```bash
python -m scripts.train_deploy     # ~35 min CPU: one proposed-fusion model per dataset
python -m scripts.check_deploy     # proves the live featuriser matches the training features
python -m scripts.check_mechanisms # proves each fusion rung computes what its name claims
python -m streamlit run app/predict_app.py
```

Then open <http://localhost:8501>.

`python -m streamlit` rather than a bare `streamlit`: the launcher only lands on `PATH` when
the environment is activated, so the bare form fails with *"'streamlit' is not recognized"* in
a fresh shell while `python -m` works from whichever interpreter you invoked. Activate the
environment first (`conda activate mpp`, or `venv\Scripts\activate` on Windows /
`source venv/bin/activate` elsewhere) and either form works.

Enter a SMILES string and get all eight properties, each with a conformal prediction set or
interval and the model's measured test score beside it. The models are trained on the
canonical DeepChem split under the same fixed setting as everything else and archived as
`deploy_proposed`, so the checkpoint, the calibration predictions and the displayed accuracy
all come from one fit.

`check_deploy` is not optional. Inference rebuilds ECFP, descriptors, graphs and ChemBERTa
embeddings from a raw string; if any of them drifts from what the pool was built with, the
model does not fail, it returns confident nonsense.

`app/streamlit_app.py` is the earlier Phase 0 app (five datasets, the baseline pipeline).
It is kept as the before side of the comparison, not superseded.

### Rebuilding the explainer PDFs

`paper/PROJECT_EXPLAINED.pdf` and `paper/MODELS_AND_DATA_EXPLAINED.pdf` need `reportlab`,
which is deliberately **not** a project dependency — nothing is installed into the environment
the archived results were produced under. Install it beside the project instead:

```bash
python -m pip install --target .pdflib reportlab
python -m scripts.make_explainer_pdf
python -m scripts.make_models_pdf
```

`.pdflib/` is gitignored; `MPP_PDFLIB` overrides the location. Every number in both PDFs is
computed from the repository at build time rather than typed.

---

## The evaluation protocol

**Two splits are always reported, never one.** The DeepChem canonical scaffold split (hard,
standard, comparable to published numbers) *and* mean ± 95% CI over five random scaffold
splits. They differ by up to **0.18 AUC on the same model and the same code**, so quoting
whichever is kinder is a way to claim almost anything.

**Comparisons are paired within split.** Splits differ in difficulty far more than models
differ from each other; pairing removes that shared variance and is the only reason a
five-point comparison has any power.

**Three levels of evidence, all reported:**

1. Per-dataset verdict from the paired difference's own 95% interval.
2. The same verdicts after a **Holm–Bonferroni correction** across the eight datasets —
   eight tests at α=0.05 produce a false positive 34% of the time, and several conclusions
   here rest on exactly one decisive dataset out of eight.
3. The **across-dataset test** (sign test, plus Wilcoxon on standardised and on raw
   differences), which is the question "does A beat B in general?" — a different question
   from the per-dataset table, and the one the project's success criterion actually named.

**Minimum detectable effect: ~±0.02 AUC / ~±0.10 RMSE.** Smaller is inside the interval and
is not a result.

**One hyper-parameter setting, held identical across every model.** That is what makes the
comparisons fair and leaves the absolute numbers untuned. Optuna is written
(`src/tune/study.py`) and deliberately never run — see that module's docstring for why
tuning one arm of a comparison would destroy the finding rather than test it.

---

## Layout

```
configs/        shared.yaml — the one hyper-parameter setting, checked against the code
scripts/        data prep, split generation, the multi-seed runner, Colab bundling
src/
  data/         split protocol, materialisation, length bucketing
  models/
    encoders/   graph (GIN / GINE / AttentiveFP), descriptor, sequence, cached
    fusion.py   concat | gated | xattn | bilinear | proposed
    multiview.py  encoders -> common 256-d projection -> fusion -> head
    heads.py    masked multi-task losses; EMBED_DIM lives here
  train/        loop.py is THE training loop; train_view / train_fusion / train_quantile
  tune/         Optuna study (written, never run — read the docstring first)
  eval/         metrics, paired stats, conformal, calibration, similarity, gate analysis
  baselines/    Chemprop (subprocess) and AttentiveFP (via the encoder interface)
  deploy/       featurize.py (SMILES -> the three views), predict.py (checkpoint -> answer)
app/            predict_app.py serves the enhanced models; streamlit_app.py the baseline ones
notebooks/      Colab notebooks for the GPU runs
paper/          draft.md, plus two plain-English PDFs explaining the project and the design
results/        runs/<variant>/{metrics,preds} is the authoritative archive
```

## Datasets

tox21 (12 tasks), bbbp (1), clintox (2), bace (1), sider (27) — classification;
esol (logS), lipophilicity (logD), freesolv (kcal·mol⁻¹) — regression.

Eight, not nine: HIV was dropped from the core protocol. Statistical power across datasets
comes from their *number*, not their size — eight puts the across-dataset signed-rank floor
at p=0.0078, which is all that is needed, and HIV would have cost ~60 h for the full
protocol without lowering it.

FreeSolv must be loaded with `dc.molnet.load_sampl`. `dc.molnet.load_freesolv` serves an
already z-scored target, and using it silently puts RMSE in no physical unit.

## Reproducing

`results/runs/<variant>/metrics/` is the authoritative record — the working copies in
`results/metrics/` are overwritten by every run. Colab results come back through
`python -m scripts.merge_colab_results <zip> --dry-run`; unzipping by hand nests the
directories in a way that reads as a mass deletion.

## Licence

MIT — see [`LICENSE`](LICENSE).

## Provenance

The baseline pipeline is our own earlier work, preserved runnable
(`src/train/train_ml.py`, `train_gnn.py`, `train_transformer.py`, `train_hybrid.py`,
`train_ensemble.py`) so that "we fixed the evaluation" is a checkable statement rather than an
assertion. Every comparison against it in the paper is reproducible from the archived
predictions in `results/runs/`.

## Citing this work

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22734878.svg)](https://doi.org/10.5281/zenodo.22734878)

If you use this code or its evaluation protocol, please cite it:

> Sharma, A. (2026). *Multi-view molecular property prediction, evaluated honestly*.
> Zenodo. https://doi.org/10.5281/zenodo.22734878

The DOI above is the **concept DOI** — it always resolves to the latest release. To cite a
specific version instead, use that release's own DOI (v1.0.0 is
[10.5281/zenodo.22734879](https://doi.org/10.5281/zenodo.22734879)).

`CITATION.cff` carries the machine-readable metadata, and GitHub renders a "Cite this
repository" box from it.

## Verifying the claims

Every headline number in `paper/draft.md` is asserted against the archives:

```bash
python -m scripts.check_configs      # the config matches the trainers
python -m scripts.check_paper        # the prose matches the archives
python -m scripts.check_deploy       # the live featuriser matches the training features
python -m scripts.check_mechanisms   # each fusion rung computes what its name claims
```

All four exit non-zero on any mismatch.
