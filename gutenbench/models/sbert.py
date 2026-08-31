from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, SentenceTransformerModel


@dataclass
class SBERTConfig(BaseConfig):
    model_id: str = field(
        default="sentence-transformers/all-mpnet-base-v2",
        metadata={
            "choices": [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-MiniLM-L12-v2",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 512


class SBERTQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)

        embeddings = self.encoder.tokenize(texts, **kwargs)
        return embeddings[0]


class SBERT(SentenceTransformerModel):

    name = "SBERT"
    config_class = SBERTConfig
    query_encoder_class = SBERTQueryEncoder

    def __init__(self, config: SBERTConfig):
        super().__init__(config)

    def make_model(self):
        model = SentenceTransformer(self.config.model_id, device=self.config.device)
        model.max_seq_length = self.config.max_length
        return model

    def format_queries(self, queries: list[str]) -> list[str]:
        return queries

    def format_documents(self, documents: list[str]) -> list[str]:
        return documents

    def tokenize(self, input_texts: list[str], **kwargs):
        embeddings = self.model.encode(
            input_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
