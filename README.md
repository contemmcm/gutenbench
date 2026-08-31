# GutenBench

Companion software for **GutenBench**, a diagnostic retrieval benchmark built on Bible
translations. It evaluates IR models ranging from classic sparse retrieval (BM25) to large
embedding-based dense retrievers, reporting results stratified by the benchmark's four
taxonomy levels (L1 lexical, L2 semantic, L3 explicit aggregation, L4 latent conceptual
bridging).

- Dataset (queries and relevance judgments): [contemmcm/gutenbench](https://huggingface.co/datasets/contemmcm/gutenbench) on Hugging Face
- Paper: *coming soon*

## Quick-start

Install the environment:

```bash
conda create -n gutenbench python=3.11
conda activate gutenbench
conda install -c conda-forge openjdk=21 -y

pip install -r requirements.txt
```

If the installed CUDA version does not match your system, you can fix it with:

```bash
# For CUDA 12.4
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

Place a Bible translation in `bibles/` (see [Bibles](#bibles) below), then run a
retriever on it. For example, BM25 on the King James Version (`bibles/KJV.csv`):

```bash
make BM25/KJV
```

This chunks the corpus into the benchmark's retrieval units, builds the index, runs the
retrieval, and prints one evaluation report per taxonomy level.

## Dataset and evaluation

The evaluation queries and relevance judgments are downloaded automatically from the
[Hugging Face dataset](https://huggingface.co/datasets/contemmcm/gutenbench) and
converted to TREC format under `dataset/` (`queries_LEVEL_{1..4}.tsv`,
`qrels_LEVEL_{1..4}.txt`), stratified by taxonomy level.

Each `make <MODEL>/<VERSION>` target evaluates the retrieval runs with `trec_eval`
(via Pyserini), reporting MAP, nDCG@10, and Recall@100 per level. Reports are written to
`reports/<VERSION>/<MODEL>/LEVEL_<n>.txt`.

## Bibles

> **Note:** GutenBench does not redistribute Bible translations. You are responsible for
> obtaining the versions you wish to evaluate and ensuring compliance with their
> respective licenses.

Place each Bible translation as a CSV file in the `bibles/` directory. Files should be
named by a unique version identifier (e.g., `KJV.csv`), which is used throughout the
pipeline to refer to that translation.

Each CSV must contain the following columns:

| Column    | Description                                                   | Required |
|-----------|---------------------------------------------------------------|----------|
| `book`    | Book abbreviation in OSIS format (e.g., `GEN`, `MAT`)        | Yes      |
| `chapter` | Chapter reference in OSIS format (e.g., `GEN.1`)             | Yes      |
| `verse`   | Verse reference in OSIS format (e.g., `GEN.1.1`)             | Yes      |
| `content` | Full text of the verse                                        | Yes      |
| `url`     | Source URL for the verse                                      | No       |

The `book`, `chapter`, and `verse` fields follow the [OSIS](https://crosswire.org/osis/)
(Open Scripture Information Standard) identifier scheme. OSIS is an XML-based standard
for encoding biblical texts, and its reference system uses dot-separated identifiers:
`BOOK.CHAPTER.VERSE` (e.g., `GEN.1.1` for Genesis chapter 1, verse 1). Book
abbreviations are standardized three-letter codes (`GEN`, `EXO`, `MAT`, `REV`, etc.).

Example row (from `bibles/KJV.csv`):

```
book,chapter,verse,content,url
GEN,GEN.1,GEN.1.1,"In the beginning God created the heaven and the earth.",https://www.bible.com/bible/1/GEN.1.1.KJV
```

## Retrievers

Evaluated retrievers: BM25, BGE, Qwen3, Nemotron, KaLM, GritLM, SBERT, InstructOR,
and InfX. Every retriever follows the same target pattern:

```bash
make [<MODEL>_MODEL_VERSION=<version>] <MODEL>/<CORPUS>
```

See [models/README.md](models/README.md) for the available model versions and settings
of each retriever.

To benchmark all evaluated retrievers (every model version) on a translation at once,
e.g. on the KJV:

```bash
./benchmark.sh KJV
```

## Adding a new retriever

The evaluation pipeline is modular: each retriever is a self-contained makefile. To add
one:

1. Create `models/<NAME>.mk` defining a `<NAME>/%` target that runs the full pipeline
   for a corpus (encode/index, search each level, evaluate with `trec_eval`) and prints
   the per-level reports. Use an existing makefile (e.g. `models/BGE.mk`) as a template.
2. Include it in the main `Makefile`: `include models/<NAME>.mk`.
3. Optionally, add the corresponding `make <NAME>/$version` call to `benchmark.sh` so it
   is part of the full benchmark run.

## Citation

```bibtex
@inproceedings{monteiro-etal-2026-gutenbench,
  title     = {{GutenBench}: A Taxonomy-Based Dataset for Evaluation of
               Information Retrieval Models},
  author    = {Monteiro, M{\'a}rcio and Senkin, Denys and
               Ravichander, Abhilasha and Kloft, Marius and Fellenz, Sophie},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2026}
}
```
