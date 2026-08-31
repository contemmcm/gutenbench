from .base import BaseModel, BaseConfig
from .bge import BGE, BGEConfig  # noqa: F401
from .deepct import DeepCT, DeepCTConfig  # noqa: F401
from .qwen3 import Qwen3, QwenConfig  # noqa: F401
from .nemotron import Nemotron, NemotronConfig  # noqa: F401
from .kalm import KaLM, KaLMConfig  # noqa: F401
from .sbert import SBERT, SBERTConfig  # noqa: F401
from .instructor import InstructOR, InstructORConfig  # noqa: F401
from .infx import InfX, InfXConfig  # noqa: F401
from .gritlm import GritLM, GritLMConfig  # noqa: F401


def get_model_class(name: str) -> type[BaseModel]:
    if name not in BaseModel.registry:
        raise ValueError(f"Unknown model: {name}")
    return BaseModel.registry[name]


def get_config_class(name: str) -> type[BaseConfig]:
    model_cls = get_model_class(name)
    return model_cls.config_class


__ALL__ = [
    "BaseEncoder",
    "EncoderConfig",
    "BGE",
    "BGEConfig",
    "DeepCT",
    "DeepCTConfig",
    "Qwen3",
    "QwenConfig",
    "Nemotron",
    "NemotronConfig",
    "KaLM",
    "KaLMConfig",
    "SBERT",
    "SBERTConfig",
    "InstructOR",
    "InstructORConfig",
    "InfX",
    "InfXConfig",
    "GritLM",
    "GritLMConfig",
]
