#!/bin/bash
#
# Runs the full model sweep (BM25, BGE, SBERT, QWEN, NEMOTRON, KALM, GritLM,
# InstructOR, InfX) against one or more bible versions, producing a report
# under reports/<version>/ for each model.
#
# Usage:
#   ./evaluate.sh <version> [<version> ...]   # one or more version IDs as args
#   ./evaluate.sh < path/to/versions.txt      # or one version ID per line on stdin
#
# Versions are the numeric bible IDs from bibles/<id>.csv.
#
# Example (run the sweep over every English bible version):
#   ./evaluate.sh < resources/bibles_english.txt
#
# Env vars:
#   CLEAR_CACHE=1   after each version, delete that version's embeddings/index
#                   cache (frees disk space; reports are kept). Scoped to the
#                   version just processed, so it's safe to run several
#                   evaluate.sh instances in parallel on different versions.
#
#   CLEAR_CACHE=1 ./evaluate.sh < resources/bibles_english.txt
#
# If a version's build fails partway through, it's skipped (its own steps
# stop early) and the script moves on to the next version rather than
# aborting the whole run. Exits non-zero at the end if any version failed.

any_failed=0

while IFS= read -r version; do
    (
        set -e

        make corpora/$version.csv
        make corpora/$version

        # BM25
        make BM25/$version

        # BGE
        make BGE/$version

        # SBERT
        make SBERT_MODEL_VERSION=all-mpnet-base-v2 SBERT/$version
        make SBERT_MODEL_VERSION=all-MiniLM-L6-v2 SBERT/$version
        make SBERT_MODEL_VERSION=all-MiniLM-L12-v2 SBERT/$version
        make SBERT_MODEL_VERSION=paraphrase-multilingual-mpnet-base-v2 SBERT/$version

        # QWEN
        make QWEN_MODEL_VERSION=Qwen3-Embedding-0.6B QWEN/$version
        make QWEN_MODEL_VERSION=Qwen3-Embedding-4B QWEN/$version
        make QWEN_MODEL_VERSION=Qwen3-Embedding-8B QWEN/$version

        # NEMOTRON
        make NEMOTRON_MODEL_VERSION=llama-embed-nemotron-8b NEMOTRON/$version
        make NEMOTRON_MODEL_VERSION=llama-nemotron-embed-1b-v2 NEMOTRON/$version

        # KALM
        make KALM/$version

        # GritLM
        make GRITLM/$version

        # InstructOR
        make INSTRUCTOR_MODEL_VERSION=instructor-base INSTRUCTOR/$version
        make INSTRUCTOR_MODEL_VERSION=instructor-large INSTRUCTOR/$version
        make INSTRUCTOR_MODEL_VERSION=instructor-xl INSTRUCTOR/$version

        # InfX
        make INFX_MODEL_VERSION=inf-retriever-v1-1.5b INFX/$version
        make INFX_MODEL_VERSION=inf-retriever-v1 INFX/$version
        make INFX_MODEL_VERSION=inf-retriever-v1-pro INFX/$version

        if [ -n "${CLEAR_CACHE:-}" ]; then
            make clean-cache/$version
        fi
    )
    if [ $? -ne 0 ]; then
        echo "!!! version $version failed, skipping to next version" >&2
        any_failed=1
    fi
done < <(if [ $# -gt 0 ]; then printf '%s\n' "$@"; else cat; fi)

exit $any_failed
