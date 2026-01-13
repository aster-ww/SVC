from __future__ import annotations

project = "SVC"
author = "Hui Wan"
release = "0.1.0"

extensions = [
    "myst_nb",
]

nb_execution_mode = "off"

templates_path = ["../_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/web_logo.png"
root_doc = "index"
