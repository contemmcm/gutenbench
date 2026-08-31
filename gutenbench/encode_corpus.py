"""
This script encodes a corpus using a specified retriever and saves the document
embeddings to disk in a Pyserini-compatible format.

Examples:

python -m gutenbench.encode_corpus --encoder Qwen3 \
    --input corpora/NKJV \
    --output .cache/embeddings/NKJV/Qwen3 \
    --batch-size 32 \
    --max-length 8192

python -m gutenbench.encode_corpus --encoder Qwen3 \
    --input corpora/NKJV \
    --output .cache/embeddings/NKJV/Qwen3 \
    --batch-size auto \
    --starting-batch-size 64 \
    --max-length 8192
"""

from dataclasses import dataclass, field as dataclass_field

from argparse_dataclass import ArgumentParser

from gutenbench import models
from gutenbench.utils import batch_size_type


@dataclass
class EncodeCorpusConfig:
    encoder: str = dataclass_field(
        metadata={
            "choices": sorted(models.BaseModel.registry.keys()),
        }
    )
    input: str
    output: str
    batch_size: int = dataclass_field(
        default=32,
        metadata={"type": batch_size_type, "metavar": "N|auto"},
    )
    field: str = "contents"
    device: str = None
    starting_batch_size: int = 64


def main():
    parser = ArgumentParser(EncodeCorpusConfig)
    args, model_args = parser.parse_known_args()

    model_cls = models.get_model_class(args.encoder)
    config_cls = models.get_config_class(args.encoder)

    config = ArgumentParser(config_cls).parse_args(model_args)

    if args.device:
        config.device = args.device

    model = model_cls(config)

    auto_batch_size = args.batch_size == "auto"
    batch_size = args.starting_batch_size if auto_batch_size else args.batch_size

    model.save_doc_embeddings(
        input=args.input,
        field=args.field,
        output=args.output,
        batch_size=batch_size,
        auto_batch_size=auto_batch_size,
    )


if __name__ == "__main__":
    main()
