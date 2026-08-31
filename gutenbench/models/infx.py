from dataclasses import dataclass, field

import torch
from sentence_transformers import SentenceTransformer

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, SentenceTransformerModel

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class InfXConfig(BaseConfig):
    model_id: str = field(
        default="infly/inf-retriever-v1-1.5b",
        metadata={
            "choices": [
                "infly/inf-retriever-v1-1.5b",
                "infly/inf-retriever-v1",
                "infly/inf-retriever-v1-pro",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 8192
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "choices": ["bfloat16", "float16", "float32"],
            "help": "Model weight dtype. bfloat16 halves VRAM vs float32.",
        },
    )


class InfXQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)

        embeddings = self.encoder.tokenize(texts, **kwargs)
        return embeddings[0]


class InfX(SentenceTransformerModel):

    name = "InfX"
    config_class = InfXConfig
    query_encoder_class = InfXQueryEncoder

    @property
    def index_name(self) -> str:
        return self.config.model_id.replace("/", "__")

    def __init__(self, config: InfXConfig):
        super().__init__(config)

    @property
    def model(self):
        """Override to skip base class .to(device); placement is handled in make_model()."""
        if not self._model:
            self._model = self.make_model()
            self._model.eval()
        return self._model

    def make_model(self):
        torch_dtype = _DTYPE_MAP.get(self.config.torch_dtype, torch.bfloat16)

        # Single-GPU: device_map pins all layers to one device.
        # No device: device_map="auto" spreads across all available GPUs.
        device = self.config.device
        device_map = {"": device} if device else "auto"

        model = SentenceTransformer(
            self.config.model_id,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch_dtype, "device_map": device_map},
        )
        model.max_seq_length = self.config.max_length
        return model

    def format_queries(self, queries: list[str]) -> list[str]:
        task = " Given a Bible query, retrieve relevant passages that answer the query"
        return [f"Instruct: {task}\nQuery: {q}" for q in queries]

    def tokenize(self, input_texts: list[str], **kwargs):
        embeddings = self.model.encode(
            input_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
