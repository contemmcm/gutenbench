import torch
import pandas as pd

from gutenbench import settings


def convert_osis_to_plain(start, end):
    """
    Convert OSIS references to plain text format.
    Examples:
        ("GEN.2.8", "GEN.2.25") -> "Genesis 2:8-25"
        ("GEN.1.1", "GEN.2.7") -> "Genesis 1:1-31; Genesis 2:1-7"
    """
    start_book, start_chapter, start_verse = start.split(".")
    end_book, end_chapter, end_verse = end.split(".")

    if start_book != end_book:
        raise ValueError(
            f"Verse ranges must stay within one book: {start!r} -> {end!r}"
        )

    start_chapter = int(start_chapter)
    start_verse = int(start_verse)
    end_chapter = int(end_chapter)
    end_verse = int(end_verse)

    if (start_chapter, start_verse) > (end_chapter, end_verse):
        raise ValueError(f"Invalid verse range: {start!r} -> {end!r}")

    book_name = settings.BOOK_NAMES[start_book]

    if start_chapter == end_chapter:
        if start_verse == end_verse:
            return f"{book_name} {start_chapter}:{start_verse}"
        return f"{book_name} {start_chapter}:{start_verse}-{end_verse}"

    parts = [
        (
            f"{book_name} {start_chapter}:{start_verse}-"
            f"{settings.VERSE_COUNTS[f'{start_book}.{start_chapter}']}"
        )
    ]

    append = parts.append
    for chapter in range(start_chapter + 1, end_chapter):
        append(
            f"{book_name} {chapter}:1-{settings.VERSE_COUNTS[f'{start_book}.{chapter}']}"
        )
    append(f"{book_name} {end_chapter}:1-{end_verse}")

    return "; ".join(parts)


def doc_ids_to_plain(doc_ids, corpus: pd.DataFrame):
    """
    Convert doc IDs to plain text references.
    Inputs:
        doc_ids: List of document IDs (e.g., [1, 2, 3])
        corpus: DataFrame containing the corpus with columns "doc_id","title","start","end","content"
    Example:
        [1, 2, 3] -> [""Genesis 1:1-31; Genesis 2:1-7", "Genesis 2:8-25", "Genesis 3:1-24"]
    """

    plain_refs = []

    for _doc_id in doc_ids:
        doc_idx = int(_doc_id.replace("doc", "")) - 1
        # fetch row in corpus by index
        row = corpus.iloc[doc_idx]
        plain_refs.append(convert_osis_to_plain(row["start"], row["end"]))

    return plain_refs


def resolve_torch_dtype(device: str, dtype: str) -> torch.dtype:
    """
    Resolve the appropriate torch dtype based on the device and dtype arguments.
    """
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def batch_size_type(value: str) -> int | str:
    """
    Argparse type converter for --batch-size: accepts a positive integer or 'auto'.
    """
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"batch-size must be a positive integer or 'auto', got: {value!r}"
        )


def load_docids(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip() for line in handle]


def write_trec_run(
    path: str, results: list[tuple[str, list[tuple[str, float]]]], tag: str
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for topic_id, hits in results:
            for rank, (docid, score) in enumerate(hits, start=1):
                handle.write(f"{topic_id} Q0 {docid} {rank} {score} {tag}\n")
