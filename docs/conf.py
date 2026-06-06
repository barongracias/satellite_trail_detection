"""Sphinx configuration for the satellite-trail detection API reference.

Heavy third-party dependencies (torch, OpenCV, astropy, …) are mocked so the
documentation builds on ReadTheDocs without a CUDA/scientific stack — autodoc
only needs to read the source and its docstrings, not execute it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable for autodoc (repo root holds src/ and scripts/).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Satellite Trail Detection"
author = "Baron Gracias"
copyright = "2026, Baron Gracias"  # noqa: A001 — Sphinx expects this name
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Mock everything third-party: with these absent, autodoc imports the project
# modules cleanly and renders their docstrings/signatures without the real deps.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchmetrics",
    "cv2",
    "numpy",
    "pandas",
    "matplotlib",
    "PIL",
    "yaml",
    "optuna",
    "scipy",
    "skimage",
    "sklearn",
    "astropy",
    "einops",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
