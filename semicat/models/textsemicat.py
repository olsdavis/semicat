"""
Text for semicat.
"""

import wandb

from semicat.models.semicat import SemicatModule


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
        self.logger.experiment.log({f"{prefix}/samples": table}, step=step)

    def on_validation_epoch_end(self) -> None:
        strings = self.sample_strings_batch(batch_size=16, sampling_method=100)
        self._log_strings(strings, prefix="val", step=self.global_step)
