# Retrievers

Every retriever follows the same target pattern:

```bash
make [<MODEL>_MODEL_VERSION=<version>] <MODEL>/<CORPUS>
```

## BM25

Classic probabilistic sparse retrieval based on term frequency and inverse document
frequency. Serves as the standard lexical baseline. Implemented via
[Pyserini](https://github.com/castorini/pyserini) / Anserini (Lucene).

```bash
make BM25/KJV
```

## BGE

Settings:

- `BGE_BATCH_SIZE`: `32`
- `BGE_MAX_LENGTH`: `512`

```bash
make BGE/KJV
```

## QWEN

Settings:

- `QWEN_MODEL_VERSION`: `Qwen3-Embedding-0.6B` (default). Available options:
  - `Qwen3-Embedding-0.6B`
  - `Qwen3-Embedding-4B`
  - `Qwen3-Embedding-8B`
- `QWEN_BATCH_SIZE`: `2`
- `QWEN_MAX_LENGTH`: `8192`

```bash
make QWEN_MODEL_VERSION=Qwen3-Embedding-0.6B QWEN/KJV
```

## NEMOTRON

Settings:

- `NEMOTRON_MODEL_VERSION`: `llama-nemotron-embed-1b-v2` (default). Available options:
  - `llama-nemotron-embed-1b-v2`
  - `llama-embed-nemotron-8b`
- `NEMOTRON_BATCH_SIZE`: `2`
- `NEMOTRON_MAX_LENGTH`: `4096`

```bash
make NEMOTRON_MODEL_VERSION=llama-nemotron-embed-1b-v2 NEMOTRON/KJV
```

## KALM

Settings:

- `KALM_MODEL_VERSION`: `KaLM-Embedding-Gemma3-12B-2511` (default)
- `KALM_BATCH_SIZE`: `2`
- `KALM_MAX_LENGTH`: `4096`

```bash
make KALM/KJV
```

## GritLM

Settings:

- `GRITLM_MODEL_VERSION`: `GritLM-7B` (default)
- `GRITLM_ENCODE_BATCH_SIZE`: `auto` (starts at 16 and adapts)
- `GRITLM_MAX_LENGTH`: `4096`
- `GRITLM_TORCH_DTYPE`: `bfloat16`

```bash
make GRITLM/KJV
```

## SBERT

Settings:

- `SBERT_MODEL_VERSION`: `all-mpnet-base-v2` (default). Available options:
  - `all-mpnet-base-v2`
  - `all-MiniLM-L6-v2`
  - `all-MiniLM-L12-v2`
  - `paraphrase-multilingual-mpnet-base-v2`
- `SBERT_BATCH_SIZE`: `32`
- `SBERT_MAX_LENGTH`: `512`

```bash
make SBERT_MODEL_VERSION=all-mpnet-base-v2 SBERT/KJV
```

## InstructOR

Settings:

- `INSTRUCTOR_MODEL_VERSION`: `instructor-base` (default). Available options:
  - `instructor-base`
  - `instructor-large`
  - `instructor-xl`
- `INSTRUCTOR_BATCH_SIZE`: `32`
- `INSTRUCTOR_MAX_LENGTH`: `512`

```bash
make INSTRUCTOR_MODEL_VERSION=instructor-base INSTRUCTOR/KJV
```

## InfX

Settings:

- `INFX_MODEL_VERSION`: `inf-retriever-v1-1.5b` (default). Available options:
  - `inf-retriever-v1-1.5b`
  - `inf-retriever-v1`
  - `inf-retriever-v1-pro`
- `INFX_BATCH_SIZE`: `4`
- `INFX_MAX_LENGTH`: `8192`

```bash
make INFX_MODEL_VERSION=inf-retriever-v1-1.5b INFX/KJV
```
