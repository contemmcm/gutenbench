from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, SentenceTransformerModel


@dataclass
class BGEConfig(BaseConfig):
    model_id: str = field(
        default="BAAI/bge-large-en-v1.5",
        metadata={
            "choices": [
                "BAAI/bge-large-en-v1.5",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 512
    prompt: str = "Represent this sentence for searching relevant passages: "


class BGEQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)

        embeddings = self.encoder.tokenize(texts, **kwargs)
        return embeddings[0]


class BGE(SentenceTransformerModel):

    name = "BGE"
    config_class = BGEConfig
    query_encoder_class = BGEQueryEncoder

    def __init__(self, config: BGEConfig):
        super().__init__(config)

    def make_model(self):
        model = SentenceTransformer(self.config.model_id, device=self.config.device)
        model.max_seq_length = self.config.max_length
        return model

    def format_queries(self, queries: list[str]) -> list[str]:
        return [f"{self.config.prompt}{query}" for query in queries]

    def format_documents(self, documents: list[str]) -> list[str]:
        return documents

    def tokenize(self, input_texts: list[str], **kwargs):
        embeddings = self.model.encode(
            input_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
