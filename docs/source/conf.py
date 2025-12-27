from __future__ import annotations

project = "SVC"
author = "Hui Wan"
release = "0.1.0"

extensions = [
    "myst_parser",
]

templates_path = ["../_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["../_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
