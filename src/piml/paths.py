"""Where things live, resolved once.

Every path a script needs comes from here, so that moving the tree or pointing at
a different corpus is one environment variable rather than an edit. Defaults are
relative to the repository root, which is three levels above this file.

    PIML_ROOT         repository root
    PIML_DATA         HDF5 corpora (gitignored; see docs/DATA.md)
    PIML_SCRATCH      intermediate arrays and solver working directories
    PIML_LOGS         the run log results.jsonl and the training traces
    PIML_MATVIZ3D     the MatViz3D clone (see docs/INSTALL.md)
    PIML_MESH_REFINE  the compiled solver driver (tools/build_mesh_refine.sh)
    PIML_CHECKPOINTS  best-validation weights, one file per run
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PIML_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("PIML_DATA", ROOT / "data"))
SCRATCH = Path(os.environ.get("PIML_SCRATCH", ROOT / "scratch"))
LOGS = Path(os.environ.get("PIML_LOGS", ROOT / "logs"))
MATVIZ3D = Path(os.environ.get("PIML_MATVIZ3D", ROOT / "MatViz3D"))
MESH_REFINE = Path(os.environ.get("PIML_MESH_REFINE", ROOT / "build" / "mesh_refine"))
CHECKPOINTS = Path(os.environ.get("PIML_CHECKPOINTS", ROOT / "checkpoints"))


def ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)
