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
from typing import Dict, List, Optional
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
      - Drops the first column (gene identifiers).
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
    # Drop gene identifier column
    if df.shape[1] > 0:
        df = df.iloc[:, 1:]
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


def weighted_confidence(
    weight_file_summand: List[str],
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Combine multiple edge confidence lists (CSV files with no header) via a weighted average.

    Each input must be provided as '<weight>*<path_to_file>'.
    Every CSV is expected to have exactly three columns **without header**:
    [source, target, confidence].

    Behavior
    --------
    - Reads each CSV as edges: (Source, Target, Confidence).
    - If a file contains duplicate (Source, Target) edges, they are averaged.
    - Unseen edges in a file are treated as Confidence = 0 for that file.
    - Computes the weighted average across files using the provided weights.
    - Returns a DataFrame with columns: 'Source', 'Target', 'Confidence' sorted descending.
    - Optionally writes the result to `output_file` (CSV with header).

    Parameters
    ----------
    weight_file_summand : List[str]
        List of weighted file specs formatted as '<weight>*<path_to_file>'.
        Example: ["0.7*/path/to/list1.csv", "0.3*/path/to/list2.csv"].
    output_file : Optional[Path], default=None
        Where to write the resulting CSV. If None, no file is written.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'Source' (str)
        - 'Target' (str)
        - 'Confidence' (float): weighted average confidence.

    Raises
    ------
    ValueError
        If an entry is invalid, a file is missing, or the weights do not sum to ~1.0 (±0.01).

    Warnings
    --------
    Issues a warning if the sum of weights slightly deviates and is normalized.
    """
    # Parse weights & file paths
    weights: List[float] = []
    files: List[Path] = []
    for spec in weight_file_summand:
        if "*" not in spec:
            raise ValueError(f"Invalid entry '{spec}'. Use '<weight>*<file_path>'.")
        w_str, f_path = spec.split("*", 1)
        try:
            w = float(w_str)
        except ValueError:
            raise ValueError(f"Invalid weight '{w_str}' in '{spec}'.")
        weights.append(w)
        files.append(Path(f_path))

    # Validate / normalize weights
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("Sum of weights must be > 0.")
    if abs(total_w - 1.0) > 0.01:
        warnings.warn(f"Weights normalized to sum to 1.0 (was {total_w:.6f}).")
        weights = [w / total_w for w in weights]

    # Read all CSVs as (Source, Target, Confidence)
    dfs: List[pd.DataFrame] = []
    for fp in files:
        if not fp.exists():
            raise ValueError(f"File not found: {fp}")
        df = pd.read_csv(
            fp,
            header=None,
            names=["Source", "Target", "Confidence"],
        )
        # Normalize types / trim whitespace
        df["Source"] = df["Source"].astype(str).str.strip()
        df["Target"] = df["Target"].astype(str).str.strip()
        df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")

        # Drop rows with missing endpoints
        df = df.dropna(subset=["Source", "Target"])
        # Treat missing/invalid confidence as 0
        df["Confidence"] = df["Confidence"].fillna(0.0)

        # If duplicates exist, average them
        df = (
            df.groupby(["Source", "Target"], as_index=False, sort=False)["Confidence"]
            .mean()
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["Source", "Target", "Confidence"])

    # Create the union of all edges (Source, Target)
    union_edges = pd.concat([d[["Source", "Target"]] for d in dfs], ignore_index=True)
    union_edges = union_edges.drop_duplicates(ignore_index=True)

    # Merge each file's confidence as a separate column
    merged = union_edges.copy()
    conf_cols: List[str] = []
    for i, df in enumerate(dfs):
        col = f"conf_{i}"
        conf_cols.append(col)
        merged = merged.merge(
            df.rename(columns={"Confidence": col}),
            on=["Source", "Target"],
            how="left",
        )

    # Fill missing confidences with 0 (edge absent in that file)
    merged[conf_cols] = merged[conf_cols].fillna(0.0)

    # Weighted average
    weighted = sum(merged[c] * weights[i] for i, c in enumerate(conf_cols))
    merged["Confidence"] = weighted

    # Keep only required columns, sort by Confidence desc
    out = merged[["Source", "Target", "Confidence"]].sort_values(
        by="Confidence", ascending=False
    ).reset_index(drop=True)

    # Optionally write CSV
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_file, index=False, header=False)

    return out
