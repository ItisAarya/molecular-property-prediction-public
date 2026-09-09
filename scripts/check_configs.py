"""
scripts/check_configs.py

Assert that `configs/shared.yaml` still describes what the code actually does.

    python -m scripts.check_configs

WHY THIS EXISTS
---------------
Every result in this project is a comparison between models trained under one fixed
hyper-parameter setting. That is the whole basis for calling the comparisons fair, and it
is stated in the plan, in PROGRESS.md and (eventually) in the paper's limitations section.

A config file that merely *records* those values would rot: someone changes a default in a
trainer, the config keeps saying the old number, and the paper's fairness claim quietly
becomes false with nothing to catch it. So this script reads the trainers' argparse
defaults directly and fails if they disagree with the config.

It deliberately does not read the config *into* the trainers. Making the config the source
of truth would mean a Colab run reproducing an old result depends on which version of a
YAML file happened to be in the bundle -- a failure mode this project has already been bitten
by once (a stale bundle silently reproducing old results). The argparse defaults stay the
source of truth; the config is the checked-in statement of what they are.

Exit code is non-zero on any mismatch, so this is usable as a pre-commit or CI gate.
"""

import argparse
import os
import sys

import yaml

CONFIG = os.path.join("configs", "shared.yaml")


def parser_defaults(module_path):
    """The argparse defaults a trainer would use if invoked with no flags."""
    import importlib

    mod = importlib.import_module(module_path)
    src = open(mod.__file__, encoding="utf-8").read()
    # Building the parser means calling main(), which trains. Re-executing just the
    # add_argument calls against a throwaway parser is the cheap way to read the defaults.
    ap = argparse.ArgumentParser()
    ns = {"ap": ap, "argparse": argparse}
    for line in _argument_lines(src):
        try:
            exec(line, {**vars(mod), **ns})
        except Exception:
            pass  # flags whose defaults reference runtime state are checked elsewhere
    return {a.dest: a.default for a in ap._actions}


def _argument_lines(src):
    """Yield each complete `ap.add_argument(...)` call, which may span several lines."""
    out, buf, depth = [], "", 0
    for line in src.splitlines():
        if buf or line.strip().startswith("ap.add_argument("):
            buf += line.strip() + " "
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                out.append(buf.strip())
                buf, depth = "", 0
    return out


CHECKS = [
    # (config path, trainer module, argparse dest)
    (("training", "epochs"), "src.train.train_view", "epochs"),
    (("training", "patience"), "src.train.train_view", "patience"),
    (("training", "batch_size"), "src.train.train_view", "batch_size"),
    (("training", "lr"), "src.train.train_view", "lr"),
    (("training", "weight_decay"), "src.train.train_view", "weight_decay"),
    (("representation", "hidden"), "src.train.train_view", "hidden"),
    (("encoders", "gine", "layers"), "src.train.train_view", "layers"),
    (("encoders", "gine", "readout"), "src.train.train_view", "readout"),
    (("encoders", "attentivefp", "timesteps"), "src.train.train_view", "timesteps"),
    (("encoders", "lora", "lora_r"), "src.train.train_view", "lora_r"),
    (("encoders", "lora", "lora_alpha"), "src.train.train_view", "lora_alpha"),
    (("encoders", "lora", "lora_dropout"), "src.train.train_view", "lora_dropout"),
    (("encoders", "lora", "pooling"), "src.train.train_view", "pooling"),
    (("training", "epochs"), "src.train.train_fusion", "epochs"),
    (("training", "patience"), "src.train.train_fusion", "patience"),
    (("training", "batch_size"), "src.train.train_fusion", "batch_size"),
    (("training", "lr"), "src.train.train_fusion", "lr"),
    (("training", "weight_decay"), "src.train.train_fusion", "weight_decay"),
    (("fusion", "rank"), "src.train.train_fusion", "rank"),
    (("fusion", "xattn_layers"), "src.train.train_fusion", "xattn_layers"),
    (("fusion", "xattn_heads"), "src.train.train_fusion", "xattn_heads"),
    (("fusion", "graph_encoder"), "src.train.train_fusion", "graph_encoder"),
    (("fusion", "seq"), "src.train.train_fusion", "seq"),
    (("representation", "hidden"), "src.train.train_fusion", "hidden"),
]


def dig(cfg, path):
    node = cfg
    for k in path:
        node = node[k]
    return node


def main():
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cache, bad, checked = {}, [], 0

    for path, module, dest in CHECKS:
        if module not in cache:
            cache[module] = parser_defaults(module)
        want = dig(cfg, path)
        got = cache[module].get(dest, "<absent>")
        checked += 1
        if isinstance(want, float) or isinstance(got, float):
            same = abs(float(want) - float(got)) < 1e-12
        else:
            same = want == got
        if not same:
            bad.append(f"  {'.'.join(path):42s} config={want!r}  {module}={got!r}")

    # Cross-module: the shared width must be shared, or the head-width confound is back.
    from src.models.heads import EMBED_DIM
    if EMBED_DIM != cfg["representation"]["embed_dim"]:
        bad.append(f"  {'representation.embed_dim':42s} config="
                   f"{cfg['representation']['embed_dim']!r}  heads.EMBED_DIM={EMBED_DIM!r}")
    checked += 1

    from src.eval.conformal import RAPS_K_REG, RAPS_LAMBDA
    for name, value, key in (("raps_lambda", RAPS_LAMBDA, "raps_lambda"),
                             ("raps_k_reg", RAPS_K_REG, "raps_k_reg")):
        checked += 1
        if value != cfg["conformal"][key]:
            bad.append(f"  {'conformal.' + key:42s} config="
                       f"{cfg['conformal'][key]!r}  conformal.py={value!r}")

    if bad:
        print(f"{len(bad)} of {checked} checks FAILED -- configs/shared.yaml no longer "
              f"describes the code:\n")
        print("\n".join(bad))
        print("\nEither the change was intended (update the config, and say so in "
              "PROGRESS.md, because it means models trained before and after are no "
              "longer comparable) or it was not (revert it).")
        sys.exit(1)

    print(f"All {checked} checks pass: configs/shared.yaml matches the trainers' defaults.")


if __name__ == "__main__":
    main()
