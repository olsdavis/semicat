"""
Defines the main module for semicat.
"""

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import lightning as L
from torchmetrics import MeanMetric


class SemicatModule(L.LightningModule):
    """
    :param net: the underlying net.
    :param prior_type: the type of prior to use, one of "gaussian" (isotropic standard Gaussian),
    "discunif" (discrete uniform).
    """

    def __init__(
        self,
        net: nn.Module,
        prior_type: Literal["gaussian", "discunif"],
        optimizer,
        scheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def prior(
        self,
        shape: tuple[int, ...],
        device: torch.device | str,
    ) -> Tensor:
        """
        Samples a point from the prior distribution.

        :param shape: The shape of the expected tensor.
        :param device: The device on which the tensor should be created.
        :return: A prior tensor of shape `shape`.
        """
        if self.hparams.prior_type == "gaussian":
            return torch.randn(shape, device=device)
        if self.hparams.prior_type == "discunif":
            cats = torch.randint(low=0, high=shape[-1], size=shape[:-1], device=device)
            return F.one_hot(cats, num_classes=shape[-1]).float()
        raise ValueError(f"unimplemented prior type `{self.hparams.prior_type}`")

    def vfm_model_step(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tensor:
        """
        VFM semi-cat step.

        :param x1: The (clean) end-point.
        :param x0: The starting point.
        :return: The loss (cross-entropy).
        """
        assert x0.shape == x1.shape
        t = torch.rand(x1.size(0), device=x1.device)
        t = t.view(-1, *((1,) * (x1.ndim - 1)))
        xt = (1.0 - t) * x0 + t * x1
        x1_pred = self.net(xt, t.view(-1))
        loss = F.cross_entropy(x1_pred, x0)
        return loss

    def model_step(
        self,
        x1: Tensor,
    ) -> Tensor:
        """
        A full semicat training step.
        
        :param x1: The target tensor, clean data.
        :return: The loss evaluated on the given data point.
        """
        # for now, only include the VFM step
        x0 = self.prior(x1.shape, device=x1.device)
        return self.vfm_model_step(x0, x1)

    def training_step(self, batch: Tensor) -> Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor) -> None:
        loss = self.model_step(batch)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_loss(loss)

    def test_step(self, batch: Tensor) -> None:
        loss = self.model_step(batch)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.test_loss(loss)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
