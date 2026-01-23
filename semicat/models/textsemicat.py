"""
Text for semicat.
"""

import wandb

import torch
from torch import Tensor

from semicat.models.semicat import SemicatModule
from semicat.metric.text_dist import TextMetrics


class TextSemicatModule(SemicatModule):
    """
    A text-specialized version of `SemicatModule`.

    :param calc_nll: Whether to calculate NLL at the end of validation epochs.
    :param nll_steps: For many how steps the model should be evaluated. If `None`,
        `[1, 2, 4, 8, 16]` by default.
    :param nll_samples: The number of samples to draw per step to evaluate the NLL.
    :param nll_model_batch_size: The batch size to use for the underlying NLL model.
    :param nll_sampling_batch_size: The batch size to use for the model's sampling.
    """

    def __init__(
        self,
        *args,
        calc_nll: bool = False,
        nll_steps: list[int] | None = None,
        nll_samples: int = 1000,
        nll_model_batch_size: int = 128,
        nll_sampling_batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.calc_nll = calc_nll
        self.nll_steps = nll_steps or [1, 2, 4, 8, 16]

    @torch.inference_mode()
    def sample_batch(
        self,
        n_samples: int,
        batch_size: int,
        *args,
        **kwargs,
    ) -> list[Tensor]:
        """
        Samples `n_samples` strings from the model, `batch_size` samples at a time.

        :param n_samples: The total number of samples to return.
        :param batch_size: The batch size.
        :param args: Additional args to `sample_batch`.
        :param kwargs: Additional kwargs to `sample_batch`.
        :return: A list of `n_samples` sampled tokens.
        """
        ret = []
        to_sample = n_samples
        while to_sample > 0:
            samples = self.sample_flow_map_batch(
                batch_size=min(batch_size, to_sample), *args, **kwargs
            )
            to_sample -= samples.size(0)
            ret.append(samples.argmax(dim=-1).to("cpu", non_blocking=True))
        return ret

    def _tokens_to_strings(self, tokens: list[Tensor]) -> list[str]:
        """
        Returns a flattened list of strings obtained from `tokens`.
        
        :param tokens: The tokens to convert.
        :return: Flattened list of strings obtained from `tokens`
        """
        ret = []

        for tok in tokens:
            strs = self.trainer.datamodule.tensor_to_strings(tok)
            ret.extend(strs)

        return ret

    def _log_strings(self, title: str, xs: list[str]):
        """
        Logs the provided strings to the logger if possible; always prints to console.
        """
        if len(xs) > 64:
            xs = xs[:64]
        if hasattr(self.logger, "experiment"):
            col = ["Text"]
            tab = wandb.Table(columns=col)
            for x in xs:
                tab.add_data(x)
            self.logger.experiment.log({title: tab}, commit=False)
        print(f"{title}: {xs}")

    def _compute_nll(
        self,
        strings: list[str],
    ) -> float:
        """
        Compute the NLL of a list of strings using a pre-trained GPT model.

        :param strings: The list of strings to compute the NLL for.
        :return: The average NLL of the strings.
        """
        return TextMetrics.compute_mean_gen_ppl(
            strings,
            self.hparams.nll_model_batch_size,
            context_size=self.in_shape[0],
        )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        for n_step in self.nll_steps:
            # A bit dirty, but might be helpful when working with limited memory
            torch.cuda.empty_cache()
            print(f"Sampling validation strings for {n_step} steps...")
            val_tokens = self.sample_batch(
                n_samples=self.hparams.nll_samples,
                batch_size=self.hparams.nll_sampling_batch_size,
                sampling_steps=n_step,
            )

            print("Computing entropy...")
            entropy = TextMetrics.compute_mean_entropy(val_tokens)
            self.log(
                f"val/entropy_{n_step}_steps",
                entropy,
                prog_bar=True,
                on_epoch=True,
                logger=True,
            )

            print("Detokenizing generated sequences")
            val_strings = self._tokens_to_strings(val_tokens)

            if self.calc_nll:
                print(f"Computing NLL for {n_step} steps...")
                nll = self._compute_nll(val_strings)
                self.log(
                    f"val/nll_{n_step}_steps",
                    nll,
                    prog_bar=True,
                    on_epoch=True,
                    logger=True,
                )

            self._log_strings(f"samples/{n_step:03d}", val_strings[:16])

        torch.cuda.empty_cache()
        print("Done text eval.")
