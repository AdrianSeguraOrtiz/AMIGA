from __future__ import annotations

from pathlib import Path
import tomllib

import amiga


def test_package_version_matches_pyproject():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert amiga.__version__ == pyproject["tool"]["poetry"]["version"]


def test_pyproject_contains_publishable_metadata():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    poetry = pyproject["tool"]["poetry"]

    assert poetry["name"] == "amiga-grn"
    assert poetry["authors"] == ["Adrián Segura-Ortiz <adrianseor.99@uma.es>"]
    assert pyproject["tool"]["poetry"]["scripts"]["amiga"] == "amiga.cli:app"

    for key in ("homepage", "repository", "documentation", "keywords", "classifiers", "license", "readme"):
        assert key in poetry
        assert poetry[key]

    assert "pareto" in poetry["keywords"]
    assert "Operating System :: OS Independent" in poetry["classifiers"]


def test_runtime_dependencies_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["tool"]["poetry"]["dependencies"]
    for dep in ("numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", "scipy"):
        assert dep in deps
