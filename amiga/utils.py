# utils.py
"""
Utility functions for AMIGA:
- File I/O helpers (pickle, JSON)
- Safe data conversion (NumPy types, NaN/inf handling)
- Expression matrix loading
- GRN weight extraction from evaluated fronts
"""

import math
from pathlib import Path
import pickle
import json
from typing import Dict
import warnings

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# File handling utilities
# ----------------------------------------------------------------------

def save_pickle(obj, path: Path) -> None:
    """Serialize an object and save it to a binary .pkl file."""
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    """Load a serialized object from a .pkl file."""
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(data, path: Path) -> None:
    """Save a Python object as a JSON file (UTF-8, human-readable)."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    """Load a JSON file and return its content as a Python object."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Data cleaning and conversion helpers
# ----------------------------------------------------------------------

def numpy_to_native(x):
    """
    Convert NumPy scalar or array types into native Python equivalents.

    Examples
    --------
    np.float32 -> float
    np.int64   -> int
    np.ndarray -> list
    """
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def nan_to_none(x):
    """
    Convert NaN or infinity values to None.
    Ensures JSON serializability and compatibility with data exports.
    """
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def clean(d: Dict):
    """
    Clean a dictionary by:
      1. Converting NumPy types to native Python types.
      2. Replacing NaN/inf values with None.

    Returns a sanitized version of the input dictionary.
    """
    return {k: nan_to_none(numpy_to_native(v)) for k, v in d.items()}


# ----------------------------------------------------------------------
# Domain-specific utilities
# ----------------------------------------------------------------------

def load_expression_matrix(csv_path: Path) -> pd.DataFrame:
    """
    Load a gene expression matrix from a CSV file and return a numeric DataFrame (genes × conditions).

    Behavior:
      - Converts non-numeric columns to numeric where possible.
      - Drops fully empty rows/columns (all NaN).
      - Ensures the resulting DataFrame contains only valid numeric entries.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing the expression matrix.

    Returns
    -------
    pd.DataFrame
        Clean numeric DataFrame (genes × conditions).
    """
    df = pd.read_csv(csv_path)
    # Attempt to coerce any remaining non-numeric columns
    df = df.apply(pd.to_numeric, errors="coerce")
    # Drop completely empty rows or columns
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def row_weights_from_front(row: pd.Series) -> Dict[str, float]:
    """
    Extract GRN file weights from a front row and normalize them if necessary.

    The function looks for columns named 'GRN_*.csv' representing the weights
    assigned to each GRN file in the evaluated front.

    Parameters
    ----------
    row : pd.Series
        Row from the evaluated front containing GRN weights.

    Returns
    -------
    Dict[str, float]
        Mapping { 'GRN_x.csv': normalized_weight }.

    Raises
    ------
    ValueError
        If no 'GRN_*.csv' columns are found or all weights are zero/NaN.

    Warnings
    --------
    Issues a warning if the extracted weights do not sum to 1.0 and are normalized.
    """
    # Identify all GRN columns
    grn_cols = [c for c in row.index if c.startswith("GRN_") and c.endswith(".csv")]
    if not grn_cols:
        raise ValueError("The evaluated front does not contain any 'GRN_*.csv' weight columns.")

    # Extract positive, non-NaN weights
    weights = {c: float(row[c]) for c in grn_cols if pd.notna(row[c]) and float(row[c]) > 0}
    s = sum(weights.values())

    # Validate weights
    if s <= 0:
        raise ValueError("All GRN_*.csv weights in this row are zero or NaN.")

    # Normalize if necessary
    if abs(s - 1.0) > 1e-3:
        warnings.warn("Weights normalized to sum to 1.0.")
        weights = {k: v / s for k, v in weights.items()}

    return weights
