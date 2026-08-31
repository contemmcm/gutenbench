from dataclasses import dataclass, field

import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer

from gutenbench.models.base import BaseConfig, SparseModel


@dataclass
class DeepCTConfig(BaseConfig):
    model_id: str = field(
        default="macavaney/deepct",
        metadata={
            "choices": [
                "macavaney/deepct",
            ],
            "help": "Hugging Face model id.",
        },
    )
    max_length: int = 512
    scale: int = 100


class DeepCT(SparseModel):
    """
    DeepCT: Context-Aware Sentence/Passage Term Importance Estimation
    (Dai & Callan, 2020).

    Predicts a scalar importance weight for each token in a document via a
    BERT + linear regression head. Weights are aggregated per word (max over
    subword tokens), scaled to integers, and the word is repeated that many
    times in the output text. The resulting weighted-term documents are indexed
    with Lucene and retrieved with BM25.

    At query time no neural encoding is required — standard BM25 is used.
    """

    name = "DeepCT"
    config_class = DeepCTConfig
    query_encoder_class = None

    def __init__(self, config: DeepCTConfig):
        super().__init__(config)

    def make_model(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.config.model_id,
            low_cpu_mem_usage=False,
        )

    def make_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.model_id)

    def format_documents(self, documents: list[str]) -> list[str]:
        return documents

    def tokenize(self, input_texts: list[str], **kwargs) -> list[str]:
        results = []
        for text in input_texts:
            words = text.split()
            if not words:
                results.append("")
                continue

            enc = self.tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            word_ids = enc.word_ids()
            enc = enc.to(self.model.device)

            with torch.inference_mode():
                # logits: [1, seq_len, 1]
                logits = self.model(**enc).logits[0, :, 0].float()  # [seq_len]

            # Max-aggregate scores over subword tokens per original word
            word_scores: dict[int, float] = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                score = float(logits[token_idx])
                if word_id not in word_scores or score > word_scores[word_id]:
                    word_scores[word_id] = score

            # Scale to integer repeat counts, build weighted text
            weighted_terms = []
            for word_id in sorted(word_scores):
                count = max(
                    0,
                    min(self.config.scale, round(word_scores[word_id] * self.config.scale)),
                )
                if count > 0 and word_id < len(words):
                    weighted_terms.extend([words[word_id]] * count)

            results.append(" ".join(weighted_terms))

        return results
