"""Sphinx configuration for the nireg documentation."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = DOC_ROOT / "generated"
NOTEBOOKS_SRC = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DST = GENERATED_ROOT / "notebooks"
README_SRC = PROJECT_ROOT / "README.md"
README_DST = GENERATED_ROOT / "README.md"


def _stage_docs_assets() -> None:
    """Copy shared repository assets into the Sphinx source tree."""
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(README_SRC, README_DST)

    if NOTEBOOKS_DST.exists():
        shutil.rmtree(NOTEBOOKS_DST)
    NOTEBOOKS_DST.mkdir(parents=True, exist_ok=True)
    for notebook in NOTEBOOKS_SRC.glob("*.ipynb"):
        notebook_data = json.loads(notebook.read_text())
        cells = notebook_data.setdefault("cells", [])
        first_cell = cells[0] if cells else None
        has_title = (
            first_cell is not None
            and first_cell.get("cell_type") == "markdown"
            and any(line.lstrip().startswith("#") for line in first_cell.get("source", []))
        )
        if not has_title:
            title = notebook.stem.replace("_", " ")
            notebook_data["cells"] = [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {title}\n"],
                },
                *cells,
            ]
        (NOTEBOOKS_DST / notebook.name).write_text(f"{json.dumps(notebook_data, indent=1)}\n")


_stage_docs_assets()
sys.path.insert(0, str(PROJECT_ROOT))

project = "nireg"
author = "Martin Reuter"
copyright = "2026, Martin Reuter"

extensions = [
    "myst_parser",
    "nbsphinx",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
numpydoc_show_class_members = False
bibtex_bibfiles = ["references.bib"]
nbsphinx_execute = "never"
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "furo"
html_title = "nireg"
html_static_path: list[str] = []
