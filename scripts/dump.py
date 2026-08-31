"""
This script converts the original corpus CSV to a folder with each entry coverted to its own JSON.

Example:
    $ python scripts/dump.py --id 1
"""

import argparse
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump the corpus in jsonl format compatible with pyserini."
    )
    parser.add_argument("--id", required=True, help="Bible version.")
    return parser.parse_args()


def doc2id(row):
    return f"doc{row.doc_id:04d}"


def main():
    args = parse_args()

    df = pd.read_csv(f"corpora/{args.id}.csv")

    output_dir = f"corpora/{args.id}"

    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        data = {
            "id": doc2id(row),
            "contents": row.content,  # used by DefaultLuceneDocumentGenerator
            "text": row.content,  # used by pyserini.encode (dense models)
        }
        ouput_file = os.path.join(output_dir, f"{doc2id(row)}.json")
        with open(ouput_file, "w", encoding="utf-8") as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
