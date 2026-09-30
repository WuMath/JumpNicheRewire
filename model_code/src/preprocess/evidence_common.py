"""Utilities shared by the evidence-enhancement preprocessing steps.

The entry scripts keep every user-editable option in a dataclass.  This file
contains only parsing, validation, matching and atomic-output helpers.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV/TSV whose HTTP client may have already removed gzip coding.

    Some browsers save a decoded CSV with the original ``.csv.gz`` filename.
    Detecting the gzip magic bytes is therefore safer than trusting the suffix.
    """

    path = Path(path)
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    return pd.read_csv(path, compression="gzip" if is_gzip else None, **kwargs)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def as_bool(values: pd.Series) -> pd.Series:
    mapping = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    lowered = values.astype("string").str.strip().str.lower()
    return lowered.map(mapping).fillna(False).astype(bool)


def canonical_numeric_id(value: object, significant_digits: int = 13) -> str | None:
    """Round a very large numeric cell ID for a controlled fallback join.

    STalign documents that some IDs changed precision after being read by R.
    Exact string matching is always attempted first.  This key is used only for
    unmatched rows, and callers must reject duplicated fallback keys.
    """

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        with localcontext() as context:
            context.prec = max(significant_digits + 4, 20)
            number = Decimal(text)
            if not number.is_finite() or number == 0:
                return None
            exponent = number.adjusted() - significant_digits + 1
            rounded = number.quantize(Decimal(1).scaleb(exponent))
            return f"{rounded:.{significant_digits - 1}E}"
    except (InvalidOperation, ValueError):
        return None


def safe_numeric_id_map(
    values: pd.Series,
    significant_digits: int,
) -> dict[str, str]:
    """Return non-ambiguous rounded-key -> original-ID mappings."""

    table = pd.DataFrame(
        {
            "original": values.astype("string").str.strip(),
            "key": values.map(lambda value: canonical_numeric_id(value, significant_digits)),
        }
    ).dropna()
    counts = table.groupby("key")["original"].nunique()
    unique_keys = set(counts[counts.eq(1)].index)
    table = table.loc[table["key"].isin(unique_keys)].drop_duplicates("key")
    return dict(zip(table["key"], table["original"]))


def write_frame_atomic(frame: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if suffix == ".parquet":
            frame.to_parquet(temporary_path, index=index)
        else:
            frame.to_csv(temporary_path, index=index)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def file_hash(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    expected_md5: str | None = None,
    overwrite: bool = False,
    timeout_seconds: int = 120,
) -> Path:
    """Download one public resource atomically and optionally verify MD5."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if expected_md5 and file_hash(destination, "md5") != expected_md5.lower():
            raise ValueError(
                f"Existing file has the wrong MD5: {destination}. "
                "Set overwrite=True only after checking the source URL."
            )
        return destination

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "JumpNicheRewire/0.1 data-preparation",
            "Accept-Encoding": "identity",
        },
    )
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if expected_md5 and file_hash(partial, "md5") != expected_md5.lower():
            raise ValueError(f"Downloaded file failed MD5 verification: {destination.name}")
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def normalized_gene_lookup(genes: Iterable[object]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for value in genes:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            lookup.setdefault(text.upper(), text)
    return lookup


def scale_zero_one(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    lower, upper = float(finite.min()), float(finite.max())
    if upper <= lower:
        return pd.Series(1.0, index=values.index, dtype=float)
    return ((numeric - lower) / (upper - lower)).clip(0.0, 1.0)


def update_summary_item(path: Path, item: str, value: object) -> None:
    """Update or append one item in a two-column project summary CSV."""

    path = Path(path)
    if path.exists():
        frame = pd.read_csv(path)
        if not {"item", "value"}.issubset(frame.columns):
            raise ValueError(f"Summary table has no item/value columns: {path}")
        if frame["item"].astype(str).eq(item).any():
            frame.loc[frame["item"].astype(str).eq(item), "value"] = value
        else:
            frame = pd.concat(
                [frame, pd.DataFrame({"item": [item], "value": [value]})],
                ignore_index=True,
            )
    else:
        frame = pd.DataFrame({"item": [item], "value": [value]})
    write_frame_atomic(frame, path)
