"""
Metrics on text distribution.

Based on Duo by Sahoo, et al. evaluation pipeline.
"""

import os
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
import transformers


class TextMetrics:
    """
    Metrics for textual data (entropy, Gen-PPL).
    """

    @classmethod
    def _load_tokenizer(cls, tokenizer_model: str = "gpt2-large"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @classmethod
    def _retokenize(cls, tokenizer: transformers.AutoTokenizer, max_length: int, text_samples: list[str], device: str) -> tuple[Tensor, Tensor]:
        tokenizer_kwargs = {
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'truncation': True,
            'padding': True,
            'max_length': max_length,
        }
        samples = tokenizer(text_samples, **tokenizer_kwargs)
        return samples['input_ids'].to(device), samples['attention_mask'].to(device)

    @classmethod
    def compute_mean_entropy(cls, tokens: list[Tensor]) -> float:
        """
        Computes the entropy as is typically done: on the token id level.
        """
        entropies = []
        for batch in tokens:
            _, counts = batch.unique(return_counts=True, sorted=False)
            entropy = torch.special.entr(
                counts.float() / counts.sum()).sum().item()
            entropies += [entropy]
        return float(np.mean(entropies))

    @classmethod
    def compute_mean_gen_ppl(
        cls,
        text_samples: list[str],
        batch_size: int,
        context_size: int = 1024,
        ppl_model: str = "gpt2-large",
        device: str = "cuda",
    ) -> float:
        """
        Computes the generative perplexity of the given samples, `samples`,
        tokenizing them with `tokenizer`, and using the perplexity of the
        specific model, `ppl_model`.
        """
        with torch.inference_mode():
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            ppls = []
            effective_size = 0
            tokenizer = TextMetrics._load_tokenizer()
            model = transformers.AutoModelForCausalLM.from_pretrained(ppl_model).eval()
            model = model.to(device)
            samples, attn_mask = TextMetrics._retokenize(tokenizer, context_size, text_samples, device)
            batch_size = min(samples.size(0), batch_size)
            n_batches = samples.size(0) // batch_size
            for i in range(n_batches):
                _samples = torch.split(
                    samples[i * batch_size: (i + 1) * batch_size],
                    context_size,
                    dim=-1,
                )
                _attn_mask = torch.split(
                    attn_mask[i * batch_size: (i + 1) * batch_size],
                    context_size,
                    dim=-1,
                )
                for (sample_chunk, attn_mask_chunk) in zip(_samples, _attn_mask):
                    logits = model(sample_chunk, attention_mask=attn_mask_chunk).logits
                    logits = logits.transpose(-1, -2)
                    nlls = F.cross_entropy(
                        logits[..., :-1],
                        sample_chunk[..., 1:],
                        reduction="none",
                    )
                    first_eos = (sample_chunk == tokenizer.eos_token_id).cumsum(-1) == 1
                    token_mask = sample_chunk != tokenizer.eos_token_id
                    valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
                    ppl = (nlls * valid_tokens).sum()
                    effective_size += valid_tokens.sum().item()
                    ppls += [ppl.cpu().numpy()]
        del model

        pre = np.exp(sum(ppls) / effective_size)
        return float(pre)
