from typing import Any
import pickle
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule


class Text8Dataset(Dataset):
    def __init__(self, all_data: torch.Tensor, k: int, dim: int):
        self.all_data = all_data
        self.k = k
        self.dim = dim

    def __len__(self) -> int:
        return len(self.all_data) - self.k + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.all_data[idx:idx + self.k]


class Text8DataModule(LightningDataModule):
    """
    Text8 data module.

    :param k: sequence length.
    :param small_run: useful for debugging.
    """

    def __init__(
        self,
        train_val_test_split: tuple[int, int, int],
        k: int = 256,
        data_dir: str = "data/text8",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        small_run: bool = False,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # corresponds to window size
        self.k = k

        self.data_dir = data_dir

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Nothing to download."""

    def setup(self, stage: str | None = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        data_dir = self.data_dir
        meta_path = os.path.join(data_dir, 'meta.pkl')
        print(f"loading meta from {meta_path}")
        assert os.path.exists(meta_path)
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.meta_vocab_size = self.meta['vocab_size']
        print(f"found vocab_size = {self.meta_vocab_size} (inside {meta_path})")

        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']

        device = self.trainer.strategy.root_device

        data_train_base = torch.from_numpy(
            np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16).astype(np.int32)
        ).long().to(device, non_blocking=True)
        data_val_base = torch.from_numpy(
            np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16).astype(np.int32)
        ).long().to(device, non_blocking=True)
        # build dataset
        trl = self.hparams.train_val_test_split[0]
        val = self.hparams.train_val_test_split[1]
        tsl = self.hparams.train_val_test_split[2]
        if self.hparams.small_run:
            trl = self.hparams.batch_size * 15
            val = 1024
            tsl = 1024
        self.data_train = Text8Dataset(data_train_base[:trl], self.k, self.meta_vocab_size)
        self.data_val = Text8Dataset(data_val_base[:val], self.k, self.meta_vocab_size)
        self.data_test = Text8Dataset(data_val_base[val:val+tsl], self.k, self.meta_vocab_size)
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def tensor_to_strings(self, tensor: torch.Tensor) -> list[str]:
        """
        Convert a tensor of indices to a list of strings.
        """
        ret = []
        for seq in tensor.tolist():
            ret += ["".join([self.itos[i] for i in seq])]
        return ret

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )


if __name__ == "__main__":
    module = Text8DataModule((351_563, 20_000, 19_063))
    module.setup()
    #print(next(iter(module.train_dataloader())))
    from semicat.metric.nll import get_gpt
    gpt = get_gpt().to("cuda")
    samples = []
    batch = next(iter(module.train_dataloader()))

    # Convert one-hot to indices
    samples = batch.argmax(dim=-1)
    indices = samples[:5]  # Take first 5 samples

    # Convert to strings
    strings = module.tensor_to_strings(indices)
    import ipdb; ipdb.set_trace()

    # Get NLL
    nll = gpt(strings)
    import ipdb; ipdb.set_trace()
