"""Sphinx configuration for the satellite-trail detection documentation.

Heavy third-party dependencies (torch, OpenCV, astropy, …) are mocked so the
documentation builds on ReadTheDocs without a CUDA/scientific stack — autodoc
only needs to read the source and its docstrings, not execute it.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
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
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

if find_spec("sphinx_copybutton") is not None:
    extensions.append("sphinx_copybutton")

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

# Overview tables on the API pages are written by hand, so no stub generation.
autosummary_generate = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# sphinx-autodoc-typehints moves the annotations into the parameter descriptions
# so signatures stay readable and types appear once, next to each argument.
autodoc_typehints = "description"
typehints_fully_qualified = False
always_document_param_types = False

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Satellite Trail Detection"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
    "style_external_links": True,
}

# "Edit on GitHub" links in the theme header.
html_context = {
    "display_github": True,
    "github_user": "barongracias",
    "github_repo": "satellite_trail_detection",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Let copy buttons skip interactive prompts so only the command is copied.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regex = True
