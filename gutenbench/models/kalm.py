from dataclasses import dataclass, field

import torch

from sentence_transformers import SentenceTransformer

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, SentenceTransformerModel


@dataclass
class KaLMConfig(BaseConfig):
    model_id: str = field(
        default="tencent/KaLM-Embedding-Gemma3-12B-2511",
        metadata={
            "choices": [
                "tencent/KaLM-Embedding-Gemma3-12B-2511",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 4096
    prompt: str = (
        "Instruct: Given a Bible query, retrieve documents that answer the query \nQuery:"
    )


class KaLMQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)

        embeddings = self.encoder.tokenize(texts, **kwargs)
        return embeddings[0]


class KaLM(SentenceTransformerModel):

    name = "KaLM"
    config_class = KaLMConfig
    query_encoder_class = KaLMQueryEncoder

    def __init__(self, config: KaLMConfig):
        super().__init__(config)

    def make_model(self):
        model = SentenceTransformer(
            self.config.model_id,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self.config.device,
        )
        model.max_seq_length = self.config.max_length
        return model

    def format_queries(self, queries: list[str]) -> list[str]:
        return [f"{self.config.prompt} {query}" for query in queries]

    def format_documents(self, documents):
        return documents

    def tokenize(self, input_texts: list[str], **kwargs):
        embeddings = self.model.encode(
            input_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
