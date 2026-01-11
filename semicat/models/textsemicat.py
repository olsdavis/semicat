"""
Text for semicat.
"""

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
        samples = self.sample_flow_map_batch(batch_size=batch_size, *args, **kwargs)
        indices = samples.argmax(dim=-1).cpu()
        return self.trainer.datamodule.tensor_to_strings(indices)

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
        Comes with added (ugly) logic for GPU inference.

        :param strings: The list of strings to compute the NLL for.
        :return: The average NLL of the strings.
        """
        gpt = get_gpt()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpt = gpt.to(self.device)
        try:
            return gpt(strings)
        finally:
            if torch.cuda.is_available():
                gpt = gpt.to("cpu")
                torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        print("Sampling strings for val NLL...")
        strings = self.sample_strings_batch(batch_size=128, sampling_steps=10)
        self._log_strings("val/samples", strings[:16])
        print("Computing NLL...")
        nll = self._compute_nll(strings)
        print("Done!")
        self.log("val/nll@128-10", nll, on_step=False, on_epoch=True, prog_bar=True)
