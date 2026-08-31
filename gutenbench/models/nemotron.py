from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from pyserini.encode import QueryEncoder

from gutenbench.models.base import TransfomerModel, BaseConfig

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class NemotronConfig(BaseConfig):
    model_id: str = field(
        default="nvidia/llama-nemotron-embed-1b-v2",
        metadata={
            "choices": [
                "nvidia/llama-nemotron-embed-1b-v2",
                "nvidia/llama-embed-nemotron-8b",
            ],
            "help": "Hugging Face model id.",
        },
    )

    max_length: int = 4096

    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "choices": ["bfloat16", "float16", "float32"],
            "help": "Model weight dtype. bfloat16 halves VRAM vs float32.",
        },
    )


class NemotronQueryEncoder(QueryEncoder):
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


class Nemotron(TransfomerModel):

    name = "Nemotron"
    config_class = NemotronConfig
    query_encoder_class = NemotronQueryEncoder

    @property
    def index_name(self) -> str:
        return self.config.model_id.replace("/", "__")

    def __init__(self, config: NemotronConfig):
        super().__init__(config)

        if self.config.model_id == "nvidia/llama-nemotron-embed-1b-v2":
            self.query_prefix = "query: "
            self.doc_prefix = "passage: "
        elif self.config.model_id == "nvidia/llama-embed-nemotron-8b":
            task_instruction = (
                "Given a Bible question, retrieve passages that answer the question"
            )
            self.query_prefix = f"Instruct: {task_instruction}\nQuery: "
            self.doc_prefix = ""  # No instruction is required for documents
        else:
            raise ValueError(f"Unknown model_id: {self.config.model_id}")

    def make_model(self):
        torch_dtype = _DTYPE_MAP.get(self.config.torch_dtype, torch.bfloat16)
        model = AutoModel.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
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
        return [f"{self.query_prefix}{query}" for query in queries]

    def format_documents(self, documents: list[str]) -> list[str]:
        return [f"{self.doc_prefix}{doc}" for doc in documents]

    def tokenize(self, input_texts: list[str], max_length: int = 4096):
        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict, use_cache=False)

        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        return embeddings.detach().cpu().float().numpy()


def average_pool(last_hidden_states, attention_mask):
    """Average pooling with attention mask."""
    last_hidden_states = last_hidden_states.to(torch.float32)
    last_hidden_states_masked = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    embedding = (
        last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    )
    embedding = F.normalize(embedding, dim=-1)
    return embedding
