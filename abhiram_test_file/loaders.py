"""
Load transliteration text from various datasets for synthetic page generation.

Supported sources:
- Oracc: row = word, document id = id_text
- Archibab: row = document, first column = doc id, line array (list of strings)
- OARE: row = document, transliteration_orig, first column = doc id
- CDLI: row = document, column 'transliteration', id_text = doc id
- Generic CSV: configurable id and text column names
"""

from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import pandas as pd


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transliteration file not found: {path}")
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace", **kwargs)


def load_oracc(csv_path: str) -> Iterator[Tuple[str, str]]:
    """
    Oracc: row = word. Document id = id_text.
    Yields (document_id, full_transliteration_text) per document.
    """
    df = _read_csv(csv_path)
    if "id_text" not in df.columns:
        raise ValueError("Oracc CSV must have column 'id_text'")
    # Assume one column is the word; aggregate by id_text
    text_col = None
    for c in ("word", "form", "transliteration", "cf", "text"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = [c for c in df.columns if c != "id_text"][0]
    grouped = df.groupby("id_text", dropna=False)
    for doc_id, grp in grouped:
        if pd.isna(doc_id):
            continue
        words = grp[text_col].astype(str).fillna("").tolist()
        text = " ".join(w.strip() for w in words if w.strip())
        yield str(doc_id), text


def load_archibab(csv_path: str) -> Iterator[Tuple[str, str]]:
    """
    Archibab: row = document. First column = document id.
    Lines stored as array e.g. ['word word', 'word word'].
    Yields (document_id, full_transliteration_text).
    """
    df = _read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("Archibab CSV must have at least 2 columns (id + lines)")
    id_col = df.columns[0]
    # Second column is typically the line array (may be string repr of list)
    lines_col = df.columns[1]
    for _, row in df.iterrows():
        doc_id = row[id_col]
        if pd.isna(doc_id):
            continue
        raw = row[lines_col]
        if isinstance(raw, str):
            import ast
            try:
                lines = ast.literal_eval(raw)
            except Exception:
                lines = [raw]
        elif isinstance(raw, list):
            lines = raw
        else:
            lines = [str(raw)] if pd.notna(raw) else []
        text = "\n".join(str(l).strip() for l in lines if str(l).strip())
        yield str(doc_id), text


def load_oare(csv_path: str) -> Iterator[Tuple[str, str]]:
    """
    OARE: row = document. transliteration_orig = text. First column = document id.
    """
    df = _read_csv(csv_path)
    if "transliteration_orig" not in df.columns:
        raise ValueError("OARE CSV must have column 'transliteration_orig'")
    id_col = df.columns[0]
    for _, row in df.iterrows():
        doc_id = row[id_col]
        if pd.isna(doc_id):
            continue
        text = row["transliteration_orig"]
        if pd.isna(text):
            continue
        yield str(doc_id), str(text).strip()


def load_cdli(csv_path: str) -> Iterator[Tuple[str, str]]:
    """
    CDLI: row = document. transliteration = text, id_text = document id.
    """
    df = _read_csv(csv_path)
    if "transliteration" not in df.columns:
        raise ValueError("CDLI CSV must have column 'transliteration'")
    id_col = "id_text" if "id_text" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        doc_id = row[id_col]
        if pd.isna(doc_id):
            continue
        text = row["transliteration"]
        if pd.isna(text):
            continue
        yield str(doc_id), str(text).strip()


def load_generic(
    csv_path: str,
    id_column: str,
    text_column: str,
) -> Iterator[Tuple[str, str]]:
    """
    Generic CSV: id_column = document id, text_column = transliteration text.
    """
    df = _read_csv(csv_path)
    if id_column not in df.columns or text_column not in df.columns:
        raise ValueError(
            f"Generic CSV must have columns '{id_column}' and '{text_column}'. "
            f"Found: {list(df.columns)}"
        )
    for _, row in df.iterrows():
        doc_id = row[id_column]
        if pd.isna(doc_id):
            continue
        text = row[text_column]
        if pd.isna(text):
            continue
        yield str(doc_id), str(text).strip()


def load_oa_published(csv_path: str) -> Iterator[Tuple[str, str]]:
    """
    OA_Published.csv: row = document, first column = document id.
    Assumes a column with transliteration content (common names: transliteration, text, content).
    """
    df = _read_csv(csv_path)
    id_col = df.columns[0]
    text_col = None
    for c in ("transliteration", "transliteration_orig", "text", "content", "line_text"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    for _, row in df.iterrows():
        doc_id = row[id_col]
        if pd.isna(doc_id):
            continue
        text = row[text_col]
        if pd.isna(text):
            continue
        yield str(doc_id), str(text).strip()


def load_finaldf(csv_path: str, id_column: str = "id_text", text_column: str = "transliteration") -> Iterator[Tuple[str, str]]:
    """
    Finaldf.csv: use generic loader; adjust id_column/text_column if different.
    """
    return load_generic(csv_path, id_column=id_column, text_column=text_column)


def load_transliteration_csv(
    csv_path: str,
    id_column: str = "id_text",
    text_column: str = "transliteration",
) -> Iterator[Tuple[str, str]]:
    """Transliteration.csv: generic id + transliteration column."""
    return load_generic(csv_path, id_column=id_column, text_column=text_column)


# Registry for notebook dropdown
SOURCES = {
    "oracc": ("Oracc (id_text, word per row)", load_oracc),
    "archibab": ("Archibab (doc per row, line array)", load_archibab),
    "oare": ("OARE (transliteration_orig)", load_oare),
    "cdli": ("CDLI (transliteration, id_text)", load_cdli),
    "oa_published": ("OA_Published (first col = id)", load_oa_published),
    "generic": ("Generic CSV (id_column, text_column)", load_generic),
    "finaldf": ("Finaldf.csv (generic)", load_finaldf),
    "transliteration": ("Transliteration.csv (generic)", load_transliteration_csv),
}


def load_all_documents(
    source: str,
    csv_path: str,
    id_column: Optional[str] = None,
    text_column: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Load all documents from a supported source. Returns list of (doc_id, text).
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Choose from: {list(SOURCES)}")
    _, loader = SOURCES[source]
    if source == "generic":
        if not id_column or not text_column:
            raise ValueError("generic source requires id_column and text_column")
        return list(loader(csv_path, id_column=id_column, text_column=text_column))
    if source in ("finaldf", "transliteration"):
        return list(loader(
            csv_path,
            id_column=id_column or "id_text",
            text_column=text_column or "transliteration",
        ))
    return list(loader(csv_path))
