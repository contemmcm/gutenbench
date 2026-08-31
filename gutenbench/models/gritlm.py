from dataclasses import dataclass, field

import torch

from gritlm import GritLM as _GritLMModel

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, SentenceTransformerModel


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _gritlm_instruction(instruction: str) -> str:
    """
    Format an instruction string for GritLM embedding.

    Queries use the task instruction; documents pass an empty string.
    The ``<|user|>`` and ``<|embed|>`` markers are special tokens the model
    was trained to recognize.  The gritlm package then masks these prefix
    tokens out of mean pooling so only content tokens contribute to the
    embedding.
    """
    if instruction:
        return f"<|user|>\n{instruction}\n<|embed|>\n"
    return "<|user|>\n<|embed|>\n"


@dataclass
class GritLMConfig(BaseConfig):
    model_id: str = field(
        default="GritLM/GritLM-7B",
        metadata={
            "choices": ["GritLM/GritLM-7B"],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 4096
    task: str = "Given a Bible query, retrieve relevant passages that answer the query"
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "choices": ["bfloat16", "float16", "float32"],
            "help": "Model weight dtype. bfloat16 halves VRAM vs float32.",
        },
    )


class GritLMQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)
        embeddings = self.encoder.tokenize(texts, **kwargs)
        return embeddings[0]


class GritLM(SentenceTransformerModel):
    """
    GritLM-7B: Generative Representational Instruction Tuning.

    Uses the official ``gritlm`` package, which applies:
    - Bidirectional (non-causal) attention during embedding inference
    - Instruction-token masking from mean pooling (only content tokens
      contribute to the embedding vector)
    - Mean pooling + L2 normalisation
    """

    name = "GritLM"
    config_class = GritLMConfig
    query_encoder_class = GritLMQueryEncoder

    def __init__(self, config: GritLMConfig):
        super().__init__(config)
        self._encoding_queries = False

    @property
    def model(self):
        """Override to skip base class .to(device); placement is handled in make_model()."""
        if not self._model:
            self._model = self.make_model()
        return self._model

    def make_model(self):
        torch_dtype = _DTYPE_MAP.get(self.config.torch_dtype, torch.bfloat16)

        # Single-GPU: device_map pins all layers to one device.
        # No device: device_map="auto" spreads across all available GPUs.
        device = self.config.device
        device_map = {"": device} if device else "auto"

        return _GritLMModel(
            self.config.model_id,
            mode="embedding",
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def format_queries(self, queries: list[str]) -> list[str]:
        self._encoding_queries = True
        return queries

    def format_documents(self, documents: list[str]) -> list[str]:
        self._encoding_queries = False
        return documents

    def tokenize(self, input_texts: list[str], max_length: int = None):
        instruction = _gritlm_instruction(
            self.config.task if self._encoding_queries else ""
        )
        return self.model.encode(
            input_texts,
            instruction=instruction,
            max_length=max_length or self.config.max_length,
        )
