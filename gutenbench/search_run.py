"""
This script runs a search using a specified retriever and saves the results to disk in 
TREC format.

Examples:

python -m gutenbench.search_run --encoder Qwen3 \
    --index .cache/indexes/NKJV/Qwen3 \
    --topics datasets/RetrievalQA/queries.tsv \
    --output .cache/search_runs/NKJV/Qwen3.RetrievalQA.trec
"""

from dataclasses import dataclass, field

from argparse_dataclass import ArgumentParser

from gutenbench import models
from gutenbench.utils import batch_size_type


@dataclass
class SearchRunConfig:
    encoder: str = field(
        metadata={
            "choices": sorted(models.BaseModel.registry.keys()),
        }
    )
    index: str
    output: str
    batch_size: int = field(
        default=32,
        metadata={"type": batch_size_type, "metavar": "N|auto"},
    )
    topics: str = None
    hits: int = 1000
    threads: int = 1
    device: str = None
    starting_batch_size: int = 128
    queries_cache: str = None


def main():
    parser = ArgumentParser(SearchRunConfig)
    args, model_args = parser.parse_known_args()

    model_cls = models.get_model_class(args.encoder)
    config_cls = models.get_config_class(args.encoder)

    config = ArgumentParser(config_cls).parse_args(model_args)

    if args.device:
        config.device = args.device

    model = model_cls(config)

    auto_batch_size = args.batch_size == "auto"
    batch_size = args.starting_batch_size if auto_batch_size else args.batch_size

    model.search_run(
        index_path=args.index,
        topics_path=args.topics,
        output_path=args.output,
        batch_size=batch_size,
        hits=args.hits,
        tag=model.name,
        threads=args.threads,
        auto_batch_size=auto_batch_size,
        queries_cache_path=args.queries_cache,
    )


if __name__ == "__main__":
    main()
