# Graph Neural Network and Transformer Fusion for Molecular Property Prediction under a Strict Evaluation Protocol

Eight MoleculeNet datasets, five molecular representations, five ways of fusing them, two
external baselines and distribution-free uncertainty on top, all measured under one protocol
across six scaffold splits, with the negative results reported.

The protocol found three things worth knowing before any architecture:

- **In ClinTox and BBBP, the way a SMILES string is written predicts the label.** Approved drugs
  are written in aromatic notation and failed ones in Kekulé notation, so a language model can
  read the answer from the spelling (notation-bit AUC 1.000 for ClinTox toxicity, 0.827 for
  BBBP permeability). Canonicalising the strings drops a frozen ChemBERTa model from 0.988 to
  0.795 AUC on ClinTox. `src/data/smiles.py` now canonicalises both datasets;
  `scripts/audit_notation.py` checks any dataset.
- **With that leak removed, fusion does not beat a fingerprint-plus-descriptor MLP**, and its
  advantage over a two-layer GIN is no longer significant across datasets.
- **Minority-class coverage under marginal conformal prediction depends on class weighting in
  training, not on the architecture**: the same descriptor MLP covers 77.9% of Tox21 actives
  with class weighting and 16.3% without it, at the same AUC.

---

## What this is

An earlier MoleculeNet pipeline of our own, rebuilt to answer one question properly: *does
combining molecular representations actually help?* Answering it required first fixing an
evaluation that could not have detected the answer either way.

**What the evaluation found in that earlier code** (each of these changed reported numbers):

- Tox21's missing-label mask was dropped, so **23.9% of canonical test labels were treated as
  negatives**.
- Regression metrics were reported in z-scored units, not logS / logD / kcal·mol⁻¹.
- Model selection ran on the **test** split.
- The meta-learner, ensemble weights, calibrators and decision thresholds were all fitted
  on a single validation split and evaluated there.
- The "frozen" ChemBERTa ran with dropout active, and computed bond features were discarded.
- Only the random forest was seeded.

**What the fixed evaluation then found** (ClinTox and BBBP on canonical SMILES; "after
correction" means Holm–Bonferroni across datasets; across-dataset p-values are sign /
Wilcoxon on Cohen's *dz* / Wilcoxon on raw differences):

| Question | Answer |
|---|---|
| Does SMILES notation leak the label? | **Yes**, in ClinTox and BBBP. Canonicalising lowers every fusion model by 0.041–0.150 AUC on those two datasets |
| Does an edge-aware graph encoder (GINE) beat the two-layer GIN? | No: 0 of 8 datasets after correction, at 5× the parameters |
| Does LoRA fine-tuning beat a frozen transformer? | Only on ESOL (Holm p = 0.0054) |
| Does the proposed fusion beat the two-layer GIN across datasets? | Not significantly: favoured on 7 of 8, p = 0.070 / 0.109 / 0.078; only FreeSolv after correction |
| Does any fusion rung beat the descriptor MLP (`desc`)? | No. `proposed` is favoured on 3 of 8 (p ≥ 0.727); `gated` and `bilinear` are worse on 7 of 8 (Wilcoxon p = 0.023) |
| Does `proposed` beat a 16.5k-parameter gate? | Favoured on 6 of 8 (Wilcoxon p = 0.039, sign p = 0.289); no dataset after correction |
| Does end-to-end LoRA adaptation beat cached embeddings? | No: favoured on 0 of 6 |
| Is the graph view needed? | No detectable loss without it (favoured on 5 of 8, p ≥ 0.250), about 13× faster; equivalence shown on only 2 of 8 |
| Does bilinear fusion beat concatenation? | On both T4 ladders (6 of 6, p = 0.031), not on the CPU ladder (6 of 8, p = 0.312), and the bilinear ranks cannot be told apart |
| Would a motif (BRICS / Murcko) view add anything? | Not measurably in a linear probe (favoured on 5 of 8, p = 0.73) |
| Does a 90% conformal guarantee cover 90% of Tox21 actives? | No: 71–80% for twelve models, 10–14% for Chemprop, the earlier transformer and the random forest |
| Does moving from CPU to GPU change results? | As much as changing the seed: about a quarter of single-split results move by more than 0.02 AUC / 0.10 RMSE; five-split means rarely do |

The paper (`paper/draft.md`, built into `paper/latex/` by `python -m scripts.make_latex`)
reports every comparison, including the ones that did not go our way. The compiled PDF is
`paper/paper.pdf`; rebuild it from `paper/latex/` (Overleaf, or `tectonic main.tex`) after
editing the draft.

---

## Quick start

```bash
conda env create -f environment.yml && conda activate mpp
python -m scripts.prep_moleculenet          # DeepChem -> CSV + npz
python -m scripts.make_graphs               # RDKit graphs (34 atom / 7 bond dims)
python -m scripts.tokenize_smiles           # ChemBERTa token ids (canonical SMILES for ClinTox, BBBP)
python -m scripts.cache_embeddings          # frozen-encoder embeddings (~25 min on a laptop CPU)
python -m scripts.build_pool                # per-split artifacts -> data/pool/
python -m scripts.make_descriptors          # 217 RDKit 2-D descriptors, raw (needs the pool)
python -m scripts.make_splits               # deepchem + seeds 0-4, audits leakage
python -m scripts.verify_prep               # assertions over every artifact
python -m src.data.materialize --variant deepchem   # write the canonical split's per-split files
```

The order matters: `make_descriptors` reads `data/pool/pool_index.json`, which `build_pool`
writes, and the checks below read per-split files that only exist once a split variant has been
materialised.

Then train and compare:

```bash
python -m scripts.run_view_multiseed --tags desc --variants deepchem seed0 seed1 seed2 seed3 seed4
python -m src.eval.view_stats --a fuse_gated --b desc --datasets tox21 bbbp clintox esol lipophilicity bace sider freesolv
```

`python -m scripts.check_configs` asserts that `configs/shared.yaml` still describes the
trainers' actual defaults. Run it before trusting a comparison.

`python -m scripts.probe_motifs` answers the question the views table invites — *why is there
no motif view?* — with a measurement rather than a schedule. Three feature sets under one
linear readout across all six splits. Over the five seeded splits, fragments alone lose on 8
of 8 datasets and adding them changes nothing measurable; on the canonical split they win 2 of 8
and adding them improves 7 of 8 numbers, with no interval to say whether that means anything.
Both are reported (Section 5.7 of the paper), and the conclusion is drawn from the convention
that has error bars.

The coverage half is the sharper half, because it holds for any motif encoder rather than only
a linear one: across all 48 dataset-split pairs, **no ring-bearing test molecule shares a
Murcko scaffold with any training molecule**, because that is what a scaffold split is for. A
scaffold-level feature is the one feature the protocol guarantees cannot transfer. Note that
the probe's `ECFP+desc` arm shares the descriptor view's features but not its model, and is
weaker — its numbers are not comparable with the view's.

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

`.pdflib/` is gitignored; `MPP_PDFLIB` overrides the location. Parameter counts and model
tallies in `MODELS_AND_DATA_EXPLAINED.pdf` are computed at build time; the results in
`PROJECT_EXPLAINED.pdf` are written into `scripts/make_explainer_pdf.py`, so re-check them
against `paper/draft.md` after any re-run.

---

## The evaluation protocol

**Two splits are always reported, never one.** The DeepChem canonical scaffold split (hard,
standard, comparable to published numbers) *and* mean ± 95% CI over five random scaffold
splits. They differ by up to **0.223 AUC on the same model and the same code**, so quoting
whichever is kinder is a way to claim almost anything.

**Comparisons are paired within split.** Splits differ in difficulty far more than models
differ from each other; pairing removes that shared variance and is the only reason a
five-point comparison has any power.

**Three levels of evidence, all reported:**

1. Per dataset, a paired t-test over the five seeded splits, with the difference's own 95% interval.
2. The same verdicts after a **Holm–Bonferroni correction** across the eight datasets —
   eight tests at α=0.05 produce a false positive 34% of the time, and several conclusions
   here rest on exactly one decisive dataset out of eight.
3. The **across-dataset test** (sign test, plus Wilcoxon on standardised and on raw
   differences), which is the question "does A beat B in general?" — a different question
   from the per-dataset table, and the one the project's success criterion actually named.

**Practical threshold: ±0.02 AUC / ±0.10 RMSE.** Smaller differences are treated as too small
to matter. The values are an empirical noise threshold taken from the earlier pipeline's
split-to-split intervals, not a power calculation. Claims of *no difference* use an equivalence
test against this threshold (`python -m scripts.equivalence`).

**One hyper-parameter setting, held identical across every model.** That is what makes the
comparisons fair and leaves the absolute numbers untuned. Optuna is written
(`src/tune/study.py`) and deliberately never run — see that module's docstring for why
tuning one arm of a comparison would destroy the finding rather than test it.

---

## Layout

```
configs/        shared.yaml — the one hyper-parameter setting, checked against the code
scripts/        data prep, split generation, the multi-seed runner, probes, Colab bundling
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
comes from their *number*, not their size. Eight puts the across-dataset sign-test floor at
p=0.0078; a ninth dataset would lower it to p=0.0039, but HIV's size would have multiplied the
compute of the full protocol (an estimated ~60 GPU-hours).

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
assertion. Every comparison against it in the paper is reproducible from the archived per-split
metrics in `results/runs/<variant>/metrics/`.

## Citing this work

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22734878.svg)](https://doi.org/10.5281/zenodo.22734878)

If you use this code or its evaluation protocol, please cite it:

> Sharma, A. (2026). *Graph Neural Network and Transformer Fusion for Molecular Property
> Prediction: What a Strict Evaluation Protocol Finds* (Version 1.0.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.22734878

The DOI above is the **concept DOI** — it always resolves to the latest release. To cite a
specific version instead, use that release's own DOI (v1.0.0 is
[10.5281/zenodo.22734879](https://doi.org/10.5281/zenodo.22734879)).

`CITATION.cff` carries the machine-readable metadata, and GitHub renders a "Cite this
repository" box from it.

## Verifying the claims

Every generated table and every checked prose number in `paper/draft.md` is asserted against
the archives:

```bash
python -m scripts.check_configs      # the config matches the trainers
python -m scripts.check_paper        # the prose matches the archives
python -m scripts.check_deploy       # the live featuriser matches the training features
python -m scripts.check_mechanisms   # each fusion rung computes what its name claims
```

All four exit non-zero on any mismatch.
