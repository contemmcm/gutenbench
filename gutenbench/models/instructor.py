from dataclasses import dataclass, field

from InstructorEmbedding import INSTRUCTOR

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, SentenceTransformerModel


@dataclass
class InstructORConfig(BaseConfig):
    model_id: str = field(
        default="hkunlp/instructor-base",
        metadata={
            "choices": [
                "hkunlp/instructor-base",
                "hkunlp/instructor-large",
                "hkunlp/instructor-xl",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 512
    query_instruction: str = "Represent the question for retrieving relevant Bible passages:"
    document_instruction: str = "Represent the Bible passage for retrieval:"


class InstructORQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)

        embeddings = self.encoder.tokenize(texts, **kwargs)
        return embeddings[0]


class InstructOR(SentenceTransformerModel):

    name = "InstructOR"
    config_class = InstructORConfig
    query_encoder_class = InstructORQueryEncoder

    def __init__(self, config: InstructORConfig):
        super().__init__(config)

    def make_model(self):
        model = INSTRUCTOR(self.config.model_id, device=self.config.device)
        model.max_seq_length = self.config.max_length
        return model

    def format_queries(self, queries: list[str]) -> list[list[str]]:
        return [[self.config.query_instruction, q] for q in queries]

    def format_documents(self, documents: list[str]) -> list[list[str]]:
        return [[self.config.document_instruction, d] for d in documents]

    def tokenize(self, input_texts: list, **kwargs):
        embeddings = self.model.encode(
            input_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
