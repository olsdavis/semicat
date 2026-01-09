"""
Text for semicat.
"""

import numpy as np
import torch

import wandb

from semicat.models.semicat import SemicatModule
from semicat.metric.nll import get_gpt


class TextSemicatModule(SemicatModule):
    """
    A text-specialized version of `SemicatModule`.
    """

    def sample_strings_batch(
        self,
        batch_size: int,
        *args,
        **kwargs,
    ) -> list[str]:
        """
        Sample a batch of strings from the model.

        :param batch_size: The batch size.
        :param args: Additional args to `sample_batch`.
        :param kwargs: Additional kwargs to `sample_batch`.
        :return: A list of sampled strings.
        """
        samples = self.sample_batch(batch_size=batch_size, *args, **kwargs)
        indices = samples.argmax(dim=-1).cpu()
        return self.trainer.datamodule.tensor_to_strings(indices)

    def _log_strings(
        self,
        strings: list[str],
        prefix: str,
        step: int,
    ) -> None:
        """
        Log a list of strings to wandb.

        :param strings: The list of strings to log.
        :param prefix: The prefix to use for logging.
        :param step: The current step.
        """
        table = wandb.Table(columns=["index", "string"])
        for i, s in enumerate(strings):
            table.add_data(i, s)
        if self.logger is not None:
            self.logger.experiment.log({f"{prefix}/samples": table}, step=step)
        else:
            print("No logger found, printing in console:")
            for i, s in enumerate(strings):
                print(f"{i}: {s}")

    def _compute_nll(
        self,
        strings: list[str],
    ) -> float:
        """
        Compute the NLL of a list of strings using a pre-trained GPT model.

        :param strings: The list of strings to compute the NLL for.
        :return: The average NLL of the strings.
        """
        gpt = get_gpt()
        return gpt(strings)

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        print("Sampling strings for val NLL...")
        strings = self.sample_strings_batch(batch_size=128, sampling_method=100)
        self._log_strings(strings[:16], prefix="val", step=self.global_step)
        print("Computing NLL...")
        nll = self._compute_nll(strings)
        print("Done!")
        self.log("val/nll@128-100", nll, on_step=False, on_epoch=True, prog_bar=True)
