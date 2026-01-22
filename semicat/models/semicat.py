"""
Defines the main module for semicat.
"""

from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchdiffeq import odeint

import lightning as L
from torchmetrics import MeanMetric

from semicat.utils.shape import view_for


class SemicatModule(L.LightningModule):
    """
    :param net: the underlying net.
    :param prior_type: the type of prior to use, one of "gaussian" (isotropic standard Gaussian),
    "discunif" (discrete uniform).
    :param sd_prop: the self-distillation proportion of the batch (between 0 and 1).
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer,
        scheduler,
        in_shape: tuple[int, ...],
        prior_type: Literal["gaussian", "discunif"],
        sd_prop: float = 0.25,
        sd_type: Literal["lag", "ecld", "semi"] = "lag",
        label_smoothing: float = 0.0,
        compile: bool = False,
        ecld: bool = False,  # unused, but left for retro-compatibility of checkpoints
    ):
        assert not ecld, "deprecated, do not use"
        super().__init__()
        torch.set_float32_matmul_precision("high")
        self.save_hyperparameters(logger=False, ignore=["net"])
        # store in_shape separately, as the type becomes messy otherwise
        # through self.hparams
        self.in_shape = tuple(in_shape)
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

    def _interpolate(
        self, x0: Tensor, x1: Tensor, t: Tensor
    ) -> Tensor:
        """
        Linear interpolation between x0 and x1 at time t.

        :param x0: The starting point, continuous.
        :param x1: The end point, categorical.
        :param t: The time (between 0 and 1), broadcastable with x0.
        :return: The interpolated point.
        """
        assert x0.ndim == x1.ndim + 1
        assert t.ndim == x0.ndim
        xt = x0 * (1.0 - t)
        xt.scatter_add_(-1, x1[..., None], t.expand_as(xt[..., :1]))
        return xt
        #xtp = (1.0 - t) * x0 + t * F.one_hot(x1, num_classes=x0.size(-1)).float()
        #assert torch.allclose(xt, xtp)

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
        t = torch.rand(x1.size(0), device=x1.device)
        t = view_for(t, x0)
        # xt = (1.0 - t) * x0 + t * x1
        xt = self._interpolate(x0, x1, t)
        x1_pred: Tensor = self.net(xt, t.view(-1), t.view(-1))
        return F.cross_entropy(
            x1_pred.transpose(-1, 1),
            x1,
            label_smoothing=self.hparams.label_smoothing,
        )

    def sd_model_step(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tensor:
        """
        Self-distillation semi-cat step.

        :param x1: The (clean) end-point.
        :param x0: The starting point.
        :return: The loss.
        """
        t = torch.rand(x1.size(0), device=x1.device)
        t = view_for(t, x0)
        s = torch.rand_like(t) * t

        # xs = (1.0 - s) * x0 + s * x1
        xs = self._interpolate(x0, x1, s)

        if self.hparams.sd_type == "ecld":
            must, dmu = torch.func.jvp(
                lambda _t: self.net(xs, s.view(-1), _t, jvp_attention=True).softmax(dim=-1),
                primals=(t.view(-1),),
                tangents=(torch.ones_like(t).view(-1),),
            )
            with torch.no_grad():
                dt = (t - s) / (1.0 - s + 1e-8)
                diff = must - xs
                xst = xs + view_for(dt, xs) * diff
                mutt = self.net(xst, t.view(-1), t.view(-1)).softmax(dim=-1)

            # dimensions through which reduce
            red_dims = tuple(range(1, len(xs.shape)))
            # calculate the ECLD loss, finally
            gamma = (t - s) / (1.0 - s + 1e-8)
            # cross-entropy with mean reduction, not KL
            div = -(mutt * must.log()).sum(dim=-1).mean()
            energy = (gamma * dmu).pow(2).sum(dim=red_dims).mean()
            # TODO: 4.0 * div + 2.0 * energy??
            return div + energy
        elif self.hparams.sd_type == "lag":
            xst, dv = torch.func.jvp(
                lambda _t: self.xst(xs, s.view(-1), _t, jvp_attention=True),
                primals=(t.view(-1),),
                tangents=(torch.ones_like(t).view(-1),),
            )
            xst = xst.detach()  # no grad
            with torch.no_grad():
                dv_target = self.net(xst, t.view(-1), t.view(-1)).softmax(dim=-1)

            return (
                (1.0 - t) * dv - dv_target + xst
            ).pow(2).sum(dim=-1).mean()
        elif self.hparams.sd_type == "semi":
            raise NotImplementedError("not implemented yet")
        raise ValueError(f"unknown sd_type: '{self.hparams.sd_type}'")

    def model_step(
        self,
        batch: Tensor | dict[str, Tensor],
    ) -> tuple[Tensor, Tensor | None]:
        """
        A full semicat training step.
        
        :param batch: The input batch, either a tensor or a dictionary containing "input_ids".
        :return: The VFM and SD losses (in this order) evaluated on the given data batch. Returns `None`
        for the SD loss if the part of the batch is zero.
        """
        if isinstance(batch, dict):
            x1 = batch["input_ids"]
        else:
            x1 = batch

        x0 = self.prior((x1.size(0), *self.in_shape), device=x1.device)
        sd_split = int(self.hparams.sd_prop * x1.size(0))
        vf_loss = self.vfm_model_step(x0[sd_split:], x1[sd_split:])
        if sd_split == 0:
            return vf_loss, None
        else:
            return vf_loss, self.sd_model_step(x0[:sd_split], x1[:sd_split])

    def vf(
        self,
        xt: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Returns the vector field from the model.

        :param xt: Current point.
        :param t: Current time.
        :return: The vector field at `(xt, t)`.
        """
        # NOTE: for now, the schedule is only linear, so:
        dr = self.net(xt, t.view(-1), t.view(-1)).softmax(dim=-1) - xt
        scale = 1.0 / (1.0 - view_for(t, xt) + 1e-8)
        return dr * scale

    def xst(
        self,
        x: Tensor,
        s: Tensor,
        t: Tensor,
        jvp_attention: bool = False,
    ) -> Tensor:
        """
        Computes the flow map `x_{s,t}`.

        :param x: The starting point.
        :param s: The starting time.
        :param t: The end time.
        :param jvp_attention: Whether to use JVP attention in the model forward pass.
        :return: The point `x_{s,t}`.
        """
        assert s.shape == t.shape
        dt = (t - s) / (1.0 - s + 1e-8)
        diff = self.net(x, s.view(-1), t.view(-1), jvp_attention=jvp_attention).softmax(dim=-1) - x
        return x + view_for(dt, x) * diff

    @torch.inference_mode()
    def sample_flow_map_batch(
        self,
        batch_size: int,
        sampling_steps: int,
        x0: Tensor | None = None,
    ) -> Tensor:
        """
        Samples a batch of data from the flow map given by the model, `x_{0,1} = net(x, 0, 1)`.

        :param batch_size: The size of the batch to sample at once.
        :param sampling_steps: The number of steps to use for the flow map approximation.
        :param x0: The starting point. If `None`, starts from a prior sample.
        :return: A batch of sampled data (still on the `R^k`, "one-hot encoded" space; `argmax`
        the last dimension to get the indices).
        """
        x = x0 or self.prior((batch_size, *self.in_shape), device=self.device)
        ts = torch.linspace(0.0, 1.0, sampling_steps+1, device=self.device)
        for s, t in zip(ts[:-1], ts[1:]):
            x = self.xst(x, s.expand((batch_size,)), t.expand((batch_size,)))
        return x

    @torch.inference_mode()
    def sample_batch_vf(
        self,
        batch_size: int,
        x0: Tensor | None = None,
        sampling_method: int | Literal["dopri5"] = 100,
        sampling_args: dict | None = None,
    ) -> Tensor:
        """
        Samples a batch of data from the vector field given by the model, `v(x, t) = net(x, t, t)`.

        :param batch_size: The size of the batch to sample at once.
        :param x0: The starting point. If `None`, starts from a prior sample.
        :param sampling_method: The sampling method: either an integer for the number
        of steps, or an ODE solver (available: "dopri5").
        :param sampling_args: Additional arguments for the sampling method. Required if
        and only if `sampling_method` is not an integer.
        :return: A batch of sampled data (still on the `R^k`, "one-hot encoded" space; `argmax`
        the last dimension to get the indices).
        """
        x = x0 or self.prior((batch_size, *self.in_shape), device=self.device)

        if isinstance(sampling_method, int):
            steps = sampling_method
            assert steps > 0
            ts = torch.linspace(0.0, 1.0, steps+1, device=self.device)
            for s, t in zip(ts[:-1], ts[1:]):
                s_fill = view_for(s.expand((batch_size,)), x)
                v = self.vf(x, s_fill.view(-1))
                x += (t - s) * v
        elif sampling_method == "dopri5":
            assert sampling_args is not None and isinstance(sampling_args, dict)
            x = cast(Tensor, odeint(
                func=self.vf,
                y0=x,
                t=torch.tensor([0.0, 1.0], device=self.device),
                method="dopri5",
                **sampling_args,
            )[-1])
        else:
            raise ValueError(f"unimplemented sampling method `{sampling_method}`")

        return x

    def training_step(self, batch: Tensor | dict[str, Tensor]) -> Tensor:
        vf_loss, sd_loss = self.model_step(batch)
        total_loss = vf_loss + sd_loss if sd_loss is not None else vf_loss
        self.train_loss(total_loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/vf_loss", vf_loss, on_step=True, on_epoch=False, prog_bar=True)
        if sd_loss is not None:
            self.log("train/sd_loss", sd_loss, on_step=True, on_epoch=False, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Tensor | dict[str, Tensor]) -> None:
        vf_loss, sd_loss = self.model_step(batch)
        total_loss = vf_loss + sd_loss if sd_loss is not None else vf_loss
        self.val_loss(total_loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/vf_loss", vf_loss, on_step=True, on_epoch=False, prog_bar=False)
        if sd_loss is not None:
            self.log("val/sd_loss", sd_loss, on_step=True, on_epoch=False, prog_bar=False)

    def test_step(self, batch: Tensor | dict[str, Tensor]) -> None:
        vf_loss, sd_loss = self.model_step(batch)
        total_loss = vf_loss + sd_loss if sd_loss is not None else vf_loss
        self.test_loss(total_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/vf_loss", vf_loss, on_step=False, on_epoch=True, prog_bar=False)
        if sd_loss is not None:
            self.log("test/sd_loss", sd_loss, on_step=False, on_epoch=True, prog_bar=False)

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

    def on_fit_start(self) -> None:
        if self.hparams.compile:
            # NOTE: Replace by
            # self.net.compile()
            # to avoid prefix
            self.net = torch.compile(self.net)

    def on_load_checkpoint(self, checkpoint):
        # For now, compiled models' weights end up being saved in
        # net._orig_mod. For retro-compatibility of checkpoints,
        # we fix the problem in the state dict
        state_dict: dict = checkpoint["state_dict"]

        prefix = "net._orig_mod."

        # If the checkpoint is compatible, ignore
        if not [k for k in state_dict.keys() if prefix in k]:
            return

        # Then, replace
        checkpoint["state_dict"] = {k.replace(prefix, "net."): v for k, v in state_dict.items()}
