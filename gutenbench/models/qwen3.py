from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from pyserini.encode import QueryEncoder

from gutenbench.models.base import BaseConfig, TransfomerModel

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class QwenConfig(BaseConfig):
    model_id: str = field(
        default="Qwen/Qwen3-Embedding-0.6B",
        metadata={
            "choices": [
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-4B",
                "Qwen/Qwen3-Embedding-8B",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 8192
    task: str = "Given a Bible query, retrieve relevant passages that answer the query"
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "choices": ["bfloat16", "float16", "float32"],
            "help": "Model weight dtype. bfloat16 halves VRAM vs float32.",
        },
    )


class QwenQueryEncoder(QueryEncoder):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, text, **kwargs):
        texts = [text]
        texts = self.encoder.format_queries(texts)

        embeddings = self.encoder.tokenize(
            texts, max_length=self.encoder.config.max_length
        )
        return embeddings[0]


class Qwen3(TransfomerModel):

    name = "Qwen3"
    config_class = QwenConfig
    query_encoder_class = QwenQueryEncoder

    def __init__(self, config: QwenConfig):
        super().__init__(config)

    def make_model(self):
        torch_dtype = _DTYPE_MAP.get(self.config.torch_dtype, torch.bfloat16)
        model = AutoModel.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.config.use_cache = False
        return model

    def make_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def format_queries(self, queries: list[str]) -> list[str]:
        task = self.config.task

        return [f"Instruct: {task}\nQuery: {query}" for query in queries]

    def format_documents(self, documents: list[str]) -> list[str]:
        return documents

    def tokenize(self, input_texts: list[str], max_length: int = 8192):
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**batch_dict, use_cache=False)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().float().numpy()


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]
