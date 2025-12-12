import os
import sys

sys.path.insert(0, os.path.abspath("../../discrimintools"))

project = "discrimintools"
author = "Duvérier DJIFACK ZEBAZE"
release = "0.1.0"
copyright = "2025, {}".format(author)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",              # Markdown
    "sphinx_copybutton",        # Bouton copier
    "sphinx_design",            # Grilles, cartes, boutons, cards
    "sphinx.ext.autodoc",       # Auto-doc Python
    "sphinx.ext.napoleon",      # Docstrings Google/NumPy
    "sphinx_autodoc_typehints", # Automatically add the types
    "sphinx.ext.viewcode",      # Add links to highlighted source code
    'sphinx.ext.autosummary',   # autosummary functions and class
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    "sphinxcontrib.email",       # for mail
    "nbsphinx"                   # to add ipynb file
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_prev_next": True,
    "logo": {"text": "discrimintools"},
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/enfantbenidedieu/discrimintools",
            "icon": "fab fa-github",   # icône GitHub Font Awesome
        }
    ]
    
}
html_static_path = ['_static']
html_logo = "_static/discrimintools.svg"
html_favicon = "_static/discrimintools.svg"

numfig = True
napoleon_use_param = False # 
numfig_format = {
    'code-block': 'Listing %s',
    'figure': 'Fig. %s',
    'section': 'Section',
    'table': 'Table %s',
}
email_automode = True # for mail
nbsphinx_execute = "never" # to generate PDF without execute ipynb