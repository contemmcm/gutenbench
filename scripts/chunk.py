"""
Example:
    $ python scripts/chunk.py --index resources/headings.csv --id 1
"""

import argparse
import csv
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk a downloaded Bible into passages given an index file."
    )
    parser.add_argument("--index", required=True, help="Path to the index file.")
    parser.add_argument("--id", required=True, help="Bible version ID.")
    return parser.parse_args()


def _merged_verse_num(verse_ref, take_min):
    """Extract a single verse number from a ref that may be merged (e.g. '1SA.28.1+1SA.28.2+1SA.28.3').
    take_min=True returns the first verse number; False returns the last.
    """
    parts = verse_ref.split("+")
    nums = [int(p.rsplit(".", 1)[1]) for p in parts]
    return min(nums) if take_min else max(nums)


def _find_boundary(df_bible, verse_ref, find_last):
    """
    Find the nearest existing verse index for a reference that may be missing.

    find_last=True  → last verse in the chapter with verse number <= target (for end refs)
    find_last=False → first verse in the chapter with verse number >= target (for start refs)

    For merged verse rows (e.g. "1SA.28.1+1SA.28.2+1SA.28.3") the first verse number is
    used when find_last=True and the last verse number when find_last=False, so that a
    merged row that covers the target verse is included as a valid candidate.

    Returns None if the chapter has no qualifying verses.
    """
    chapter = verse_ref.rsplit(".", 1)[0]      # e.g. "MAT.17"
    verse_num = int(verse_ref.rsplit(".", 1)[1])  # e.g. 21

    chapter_mask = df_bible["verse"].str.startswith(chapter + ".")
    chapter_df = df_bible[chapter_mask].copy()
    if chapter_df.empty:
        return None

    chapter_df["_vnum"] = chapter_df["verse"].apply(
        lambda v: _merged_verse_num(v, take_min=find_last)
    )
    if find_last:
        candidates = chapter_df[chapter_df["_vnum"] <= verse_num]
        return candidates.index[-1] if not candidates.empty else None
    else:
        candidates = chapter_df[chapter_df["_vnum"] >= verse_num]
        return candidates.index[0] if not candidates.empty else None


# Verses whose number exceeds the chapter length in Catholic/Hebrew versification.
# Maps Protestant ref → equivalent ref used by those translations.
_VERSIFICATION_ALTERNATES = {
    "JON.1.17": "JON.2.1",   # Catholic/Hebrew: JON.1 ends at v16
    "HOS.11.12": "HOS.12.1", # Catholic/Hebrew: HOS.11 ends at v11
}


def extract(df_bible, start, end):
    """
    Concatenates the content of "verse" col between start and end (inclusive).

    If a boundary verse is absent (e.g. a textually disputed verse omitted by
    some translations), the nearest existing verse in the same chapter is used
    instead, and a warning is printed.

    Returns (content, warnings) where warnings is the number of boundaries fixed.
    """
    warnings = 0

    start_row = df_bible[df_bible["verse"] == start]
    if start_row.empty:
        alt = _VERSIFICATION_ALTERNATES.get(start)
        if alt is not None:
            start_row = df_bible[df_bible["verse"] == alt]
            if not start_row.empty:
                print(f"Warning: '{start}' not found, using versification alternate '{alt}' as start")
                warnings += 1
    if start_row.empty:
        start_idx = _find_boundary(df_bible, start, find_last=False)
        if start_idx is None:
            raise ValueError(f"Start reference not found and no fallback in chapter: {start}")
        print(f"Warning: '{start}' not found, using '{df_bible.at[start_idx, 'verse']}' as start")
        warnings += 1
    else:
        start_idx = start_row.index[0]

    end_row = df_bible[df_bible["verse"] == end]
    if end_row.empty:
        end_idx = _find_boundary(df_bible, end, find_last=True)
        if end_idx is None:
            raise ValueError(f"End reference not found and no fallback in chapter: {end}")
        print(f"Warning: '{end}' not found, using '{df_bible.at[end_idx, 'verse']}' as end")
        warnings += 1
    else:
        end_idx = end_row.index[0]

    if start_idx > end_idx:
        raise ValueError(f"Start reference comes after end reference: {start} > {end}")

    return "".join(df_bible.loc[start_idx:end_idx, "content"].tolist()), warnings


def main():
    args = parse_args()

    df_bible = pd.read_csv(f"bibles/{args.id}.csv")
    df_index = pd.read_csv(args.index)
    df_index["content"] = ""

    total_warnings = 0
    skipped = 0
    for i, row in df_index.iterrows():
        try:
            content, warnings = extract(df_bible, row["start"], row["end"])
        except ValueError as e:
            print(f"Warning: skipping passage '{row['title']}' ({row['start']}–{row['end']}): {e}")
            df_index.at[i, "content"] = " "
            skipped += 1
            continue
        df_index.at[i, "content"] = content
        total_warnings += warnings

    print(f"Boundaries fixed: {total_warnings}, passages skipped: {skipped}")

    out_file = f"corpora/{args.id}.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_index.to_csv(out_file, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()
