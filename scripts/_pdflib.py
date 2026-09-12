"""
scripts/_pdflib.py

Make `reportlab` importable for the two PDF builders, without installing it into the project
environment.

    import scripts._pdflib  # noqa: F401  -- must precede any reportlab import

WHY THIS EXISTS
---------------
`reportlab` is needed by exactly two scripts that build explainer PDFs. It is not a dependency
of the pipeline, and this project does not install anything into the environment every result
was produced under -- the Chemprop episode is the precedent: it pins its own torch, and letting
it into the project venv would have moved the versions underneath 1,254 archived fits.

So reportlab is installed to a side directory instead:

    python -m pip install --target <dir> reportlab

Both builders previously hard-coded one absolute path to such a directory on one machine. That
made the PDFs unbuildable by anyone else -- a committed script importing from a temporary
folder that does not exist outside the machine that wrote it. This resolves it in order:

1. reportlab already importable (someone installed it normally) -- use that.
2. `$MPP_PDFLIB` if set -- the documented way to point at a `--target` directory.
3. `.pdflib/` in the project root -- the default the message below suggests.

and otherwise raises with the command to run, rather than an ImportError three frames deeper.
"""

import os
import sys

ENV_VAR = "MPP_PDFLIB"
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".pdflib")


def _importable():
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def ensure():
    """Put a reportlab installation on `sys.path`, or explain how to make one."""
    if _importable():
        return

    for candidate in (os.environ.get(ENV_VAR), DEFAULT_DIR):
        if candidate and os.path.isdir(candidate):
            sys.path.insert(0, candidate)
            if _importable():
                return
            sys.path.pop(0)

    raise SystemExit(
        "reportlab is not available, and it is deliberately not a project dependency.\n\n"
        "Install it to a side directory (NOT into the project environment, which pins the\n"
        "versions every archived result was produced under):\n\n"
        f'    python -m pip install --target "{DEFAULT_DIR}" reportlab\n\n'
        f"or set {ENV_VAR} to a directory that already has it."
    )


ensure()
