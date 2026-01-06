"""
Text for semicat.
"""

import wandb

from semicat.models.semicat import SemicatModule


class TextSemicatModule(SemicatModule):
    """
    A text-specialized version of `SemicatModule`.
    """

    def _log_strings(
        self,
        strings: list[str],
        prefix: str,
        step: int,
    ) -> None:
        """
        Log a list of strings.

        :param strings: The list of strings to log.
        :param prefix: The prefix to use for logging.
        :param step: The current step.
        """
        table = wandb.Table(columns=["index", "string"])
        for i, s in enumerate(strings):
            table.add_data(i, s)
        self.logger.experiment.log({f"{prefix}_strings": table}, step=step)
