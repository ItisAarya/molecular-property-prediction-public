"""
scripts/check_mechanisms.py

Verify that each fusion rung computes the kind of function it claims to.

    python -m scripts.check_mechanisms

WHY THIS EXISTS
---------------
§5 reports that two proposed mechanisms do not pay. That conclusion is only worth reading if
the mechanisms were implemented correctly in the first place -- a silently additive "bilinear"
block would produce the same negative result for an entirely uninteresting reason. The draft
claimed this had been checked; no script and no archive backed the claim, which in a paper
whose argument is that every number should be machine-checked is exactly the wrong place to
leave an assertion.

THE TEST
--------
For a function of three views, the second-order interaction between views a and b is the
mixed difference

    D = f(a, b, c) - f(a', b, c) - f(a, b', c) + f(a', b', c)

If f is additive in its views -- f(a,b,c) = g(a) + h(b) + k(c) -- every term cancels and D is
exactly zero, whatever g, h and k are. A non-zero D is proof that the module mixes views
multiplicatively rather than merely stacking them.

So `concat` must give D = 0 to machine precision (it is the additive control at the bottom of
the ladder, and the ladder's logic depends on that being true rather than assumed), and every
other rung must give D far from zero.

Exit code is non-zero if any rung fails its expectation, so this belongs in the same gate as
check_configs and check_paper.
"""

import sys

import numpy as np
import torch

from src.models.fusion import build_fusion

D_MODEL = 256
N_VIEWS = 3
BATCH = 8
# Machine-precision zero for float32 sums over a 256-wide vector. Anything at or below this
# is cancellation; the bilinear and attention rungs come out ten orders of magnitude above it.
ADDITIVE_TOL = 1e-5

EXPECTED = {
    "concat": "additive",
    "gated": "interacting",
    "xattn": "interacting",
    "bilinear": "interacting",
    "proposed": "interacting",
}


def interaction(module, seed=0):
    """The mixed second-order difference for views 0 and 1, held at eval with fixed input."""
    torch.manual_seed(seed)
    module.eval()

    a = torch.randn(BATCH, D_MODEL)
    a2 = torch.randn(BATCH, D_MODEL)
    b = torch.randn(BATCH, D_MODEL)
    b2 = torch.randn(BATCH, D_MODEL)
    c = torch.randn(BATCH, D_MODEL)

    with torch.no_grad():
        f = lambda x, y: module([x, y, c])
        d = f(a, b) - f(a2, b) - f(a, b2) + f(a2, b2)
    return float(d.abs().max())


def main():
    print("Second-order interaction between two views, |D|max over a batch of "
          f"{BATCH}. Additive fusion must give 0.\n")
    print(f"{'rung':<12}{'|D| max':>14}  {'expected':<13}{'verdict'}")
    bad = []
    for mode, expect in EXPECTED.items():
        module = build_fusion(mode, n_views=N_VIEWS, d=D_MODEL)
        d = interaction(module)
        is_additive = d <= ADDITIVE_TOL
        ok = (expect == "additive") == is_additive
        if not ok:
            bad.append(mode)
        print(f"{mode:<12}{d:>14.3e}  {expect:<13}{'ok' if ok else 'FAILED'}")

    print()
    if bad:
        print(f"FAILED: {', '.join(bad)} does not compute the kind of function it claims.")
        print("A negative result from a broken module is not a negative result.")
        sys.exit(1)

    print("Every rung computes the kind of function it claims: `concat` is additive to "
          "machine precision,\nand every other rung mixes views multiplicatively.")


if __name__ == "__main__":
    main()
