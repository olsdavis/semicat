"""Enables NLL-based eval using a pre-trained GPT model."""

import numpy as np
import torch
from torch import nn
from transformers import GPTJForCausalLM, AutoTokenizer


class GptNll(nn.Module):
    """Holds an instance of a GPT model and tokenizer to calculate NLL."""

    def __init__(self):
        super().__init__()
        self.model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6b",
            revision="float16",
            torch_dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

    @torch.no_grad()
    def forward(self, samples: list[str]) -> float:
        """
        Calculates the average NLL of the samples.
        """
        losses = []
        for sample in samples:
            input_ids = self.tokenizer(sample, return_tensors="pt").input_ids
            output = self.model(
                input_ids,
                return_dict=True,
                labels=input_ids,
            )
            losses.append(output.loss.item())
        return np.mean(losses)


_gpt: GptNll | None = None


def get_gpt() -> GptNll:
    """
    Returns a GptNll instance.
    """
    global _gpt

    if _gpt is None:
        _gpt = GptNll()

    return _gpt
