"""
src/tune/study.py

The Optuna study named in `02_ENHANCEMENT_PLAN.md` section 4.3.

    python -m src.tune.study --mode gated --datasets bbbp --trials 30

STATUS: WRITTEN, NEVER RUN. READ THIS BEFORE RUNNING IT.
--------------------------------------------------------
Every number in this repository comes from one hyper-parameter setting, held identical
across every model (`configs/shared.yaml`, enforced by `scripts/check_configs.py`). That is
not an oversight to be corrected by running this file -- it is what makes the comparisons
mean anything. A tuned `proposed` against an untuned `desc` would measure search budget, not
architecture, and the project's central finding is that a 16.5k-parameter gate matches a
1.17M-parameter fusion block. That finding would be destroyed, not tested, by tuning one
side of it.

So there are exactly two defensible ways to use this module, and one indefensible one:

**Defensible (1): tune every model equally.** Run the same budget for every tag in the
results table, on validation only, and report the tuned table *alongside* the fixed-setting
table. Cost is the entire evaluation multiplied by `--trials`; at 8.5 h for one fusion
ladder that is days of GPU time, which is why it has not been done.

**Defensible (2): tune nothing, and state it.** Report absolute numbers as untuned, note
that every model shares one setting, and let the comparison carry the paper. This is what
the project currently does, and it is the honest reading of a small-data benchmark where
Phase 1 found four encoder upgrades that do not replicate.

**Indefensible: tune the proposed model and compare it to fixed-setting baselines.** This is
the most common way a fusion paper manufactures a win, and this project exists partly to
document the ways that happens. Do not do it here.

WHAT IT SEARCHES
----------------
The plan lists: embedding width `d`, bilinear rank `r`, cross-attention layers and heads,
dropout, learning rate, LoRA rank, and frozen-vs-end-to-end. `d` is deliberately **not**
searched: it is the common projection width that removes the head-capacity confound
(Sessions 9-10), so letting a trial change it reintroduces exactly the bug that was fixed.
Frozen-vs-e2e is not searched either -- Phase 2 measured it directly across all 8 datasets
and found 0 improvements for ~11 GPU-hours, which is a stronger answer than a search would
give.

THE OBJECTIVE IS VALIDATION, NEVER TEST
---------------------------------------
`objective()` returns the validation metric. Selecting on test is the single largest bug
Phase 0 found in the inherited pipeline (`select_winners` chose on test), and a tuner is the
easiest place in a codebase to reintroduce it. The test split is materialised, because the
loaders are built per split from one call, but it is never scored and no trial value can
depend on it -- `objective` computes metrics on `valid` only.
"""

import argparse
import json
import os

MET_DIR = os.path.join("results", "metrics")
TUNE_DIR = os.path.join("results", "tuning")


def build_space(trial, mode):
    """The search space, as a plain dict so a trial is reproducible from its record."""
    space = {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    if mode in ("bilinear", "proposed"):
        space["rank"] = trial.suggest_categorical("rank", [16, 32, 64, 128])
    if mode in ("xattn", "proposed"):
        space["xattn_layers"] = trial.suggest_int("xattn_layers", 1, 4)
        space["xattn_heads"] = trial.suggest_categorical("xattn_heads", [2, 4, 8])
    return space


def objective(trial, ds, mode, variant, epochs, patience):
    """
    One trial: train on `variant`'s training split, score on its validation split.

    The training loop is written out here rather than calling `src/train/loop.py`. That
    loop writes predictions, metrics CSVs and checkpoints under the run's tag, which is
    correct for a real run and wrong for a search -- thirty trials would leave thirty
    overwrites of the same result files and no way to tell afterwards which run produced
    the archived numbers. Nothing here writes outside `results/tuning/`.

    Imports are local so that `--help` and the guard in `main` work in an environment
    without torch, and so that reading this module costs nothing.
    """
    import torch

    from src.data.materialize import active_variant, materialize
    from src.eval.metrics import cls_metrics, is_classification, reg_metrics
    from src.models.heads import masked_bce, masked_mse, pos_weight_from_labels
    from src.models.multiview import MultiViewModel
    from src.train.train_fusion import (SPLITS, VIEW_ORDER, build_encoders,
                                        build_loaders, load_split, unpack)
    from src.train.loop import to_device
    from src.utils.seed import set_seed

    if active_variant() != variant:
        materialize(variant, verbose=False)

    seed = set_seed()
    cls = is_classification(ds)
    params = build_space(trial, mode)
    active = tuple(VIEW_ORDER)

    parts = {s: load_split(ds, s, "cached", active) for s in SPLITS}
    loaders, train_sampler = build_loaders(parts, params["batch_size"], "cached",
                                           seed, active)
    y = {s: parts[s]["y"] for s in SPLITS}

    encoders = build_encoders(parts, "cached", "gine", 256, params["dropout"], active)
    kwargs = {}
    if "rank" in params:
        kwargs["rank"] = params["rank"]
    if "xattn_layers" in params:
        kwargs["n_layers"] = params["xattn_layers"]
        kwargs["n_heads"] = params["xattn_heads"]
    model = MultiViewModel(encoders, mode=mode, n_tasks=y["train"].shape[1],
                           d=256, dropout=params["dropout"], **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    pos_w = pos_weight_from_labels(y["train"]).to(device) if cls else None

    best, bad = float("-inf"), 0
    for ep in range(1, epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        for batch in loaders["train"]:
            inputs, yb, _ = unpack(batch)
            opt.zero_grad()
            out = model(to_device(inputs, device))
            yb = yb.to(device)
            loss = masked_bce(out, yb, pos_w) if cls else masked_mse(out, yb)
            loss.backward()
            opt.step()

        # Validation only. The test split is loaded (row alignment is built per split) but
        # is never scored here, and no trial value can depend on it.
        model.eval()
        n_rows = len(parts["valid"]["smiles"])
        pred = torch.zeros(n_rows, y["valid"].shape[1])
        with torch.no_grad():
            for batch in loaders["valid"]:
                inputs, _, idx = unpack(batch)
                p = model(to_device(inputs, device)).cpu()
                pred[torch.as_tensor(idx)] = p
        p = torch.sigmoid(pred).numpy() if cls else pred.numpy()
        m = cls_metrics(y["valid"].numpy(), p) if cls else reg_metrics(
            y["valid"].numpy(), p, ds)
        # Optuna maximises, so validation RMSE is negated and both task types agree.
        value = m["auc"] if cls else -m["rmse"]

        if value > best + 1e-6:
            best, bad = value, 0
        else:
            bad += 1
            if bad >= patience:
                break
        trial.report(best, ep)

    trial.set_user_attr("params", params)
    return best


def main():
    ap = argparse.ArgumentParser(description="Optuna study (see the module docstring).")
    ap.add_argument("--mode", default="gated",
                    choices=["concat", "gated", "xattn", "bilinear", "proposed"])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--variant", default="seed0",
                    help="tune on ONE split; tuning across all five leaks the "
                         "split-to-split variance the CIs are meant to measure")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--i-have-read-the-docstring", action="store_true",
                    help="required: this study is not safe to run casually")
    args = ap.parse_args()

    if not args.i_have_read_the_docstring:
        raise SystemExit(__doc__.strip() + "\n\nRe-run with --i-have-read-the-docstring.")

    import optuna

    from src.data.materialize import active_variant, materialize

    # Restore whatever split was materialised before this ran. A study switches `data/` to
    # its tuning variant, and leaving it switched means the next command someone runs -- a
    # smoke test, an eval -- silently reads a different split than they think. The runner
    # scripts already restore; this one has to as well.
    restore = active_variant()

    os.makedirs(TUNE_DIR, exist_ok=True)
    for ds in args.datasets:
        study = optuna.create_study(direction="maximize",
                                    study_name=f"{ds}_{args.mode}",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t: objective(t, ds, args.mode, args.variant, args.epochs, args.patience),
            n_trials=args.trials)

        out = os.path.join(TUNE_DIR, f"{ds}_{args.mode}_study.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": ds, "mode": args.mode, "variant": args.variant,
                "n_trials": len(study.trials),
                "best_value": study.best_value,
                "best_params": study.best_params,
                "note": "validation score only; test was never loaded",
            }, f, indent=2)
        print(f"{ds}/{args.mode}: best validation {study.best_value:.4f} -> {out}")

    if restore and restore != args.variant:
        materialize(restore, verbose=False)
        print(f"restored active split: {restore}")


if __name__ == "__main__":
    main()
