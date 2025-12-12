# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r",encoding="utf-8") as fh:
    long_description = fh.read()

# Setting up
setuptools.setup(
    name="discrimintools",
    version="0.1.0",
    author="Duvérier DJIFACK ZEBAZE",
    author_email="djifacklab@gmail.com",
    description="Python package dedicated to Discriminant Analysis (DA) distributed under the MIT License",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/enfantbenidedieu/discrimintools",
    packages=setuptools.find_packages(where="."),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "statsmodels>=0.14.6",
        "scikit-learn>=1.8.0",
        "openpyxl>=3.1.5",
        "tabulate>=0.9.0",
        "plotnine>=0.15.1",
        "adjustText>=1.3.0"
    ],
    include_package_data=True,
    package_data={
        "": [
            "data/*",              
        ],
    },
    keywords="linear discriminant analysis, quadratic discriminant analysis, disqual, dismix, dica",
    project_urls={
        "Bug Reports": "https://github.com/enfantbenidedieu/discrimintools/issues",
        "Source": "https://github.com/enfantbenidedieu/discrimintools",
        "Documentation": "https://discrimintools.readthedocs.io",
    },
    
)