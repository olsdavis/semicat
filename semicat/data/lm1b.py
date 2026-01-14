"""
LM1B/OpenWebText DataModule.
"""
import re
from lightning import LightningDataModule
import datasets
import transformers
import torch
from torch.utils.data import DataLoader


def lm1b_detokenizer(batch: dict[str, str]) -> dict[str, list[str]]:
    ret = []
    for x in batch["text"]:
        x = x.replace("http : / / ", "http://")
        x = x.replace("https : / / ", "https://")
        x = re.sub(r" \'(\w+)", r"'\1", x)
        x = re.sub(r" (\w+) \. ", r" \1. ", x)
        x = re.sub(r" (\w+) \.$", r" \1.", x)
        x = x.replace(" ? ", "? ")
        x = re.sub(r" \?$", "?", x)
        x = x.replace(" ! ", "! ")
        x = re.sub(r" \!$", "!", x)
        x = x.replace(" , ", ", ")
        x = x.replace(" : ", ": ")
        x = x.replace(" ; ", "; ")
        x = x.replace(" / ", "/")
        x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
        x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
        x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
        x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
        x = x.replace("$ ", "$")
        x = x.replace("£ ", "£")

        ret += [x]
    return {"text": ret}


class _AlignDataset(torch.utils.data.IterableDataset):
    """
    Adjusts the dataset to the right block size.
    """

    def __init__(self, dataset, tokenizer, block_size: int, shuffle: bool):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        buf = []
        to_take = self.block_size - 2
        for ex in self.dataset:
            ids = self.tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            # emit as many full chunks as possible
            while len(buf) >= to_take:
                chunk = [self.eos] + buf[:to_take] + [self.eos]
                assert len(chunk) == self.block_size
                buf = buf[to_take:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.block_size, dtype=torch.long),
                }


class LM1BDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        max_length: int = 1024,
        dataset: str = "owt",
    ):
        assert dataset in ["lm1b", "owt"], f"unsupported dataset '{dataset}'"
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.tokenizer = None

    def _load_tokenizer(self):
        if self.hparams.dataset == "lm1b":
            self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # no vocab resize


    def _load_dataset(self, split: str):
        assert self.tokenizer is not None, "need tokenizer"
        # cache_dir = None if "DATA" not in os.environ else os.path.join(os.environ["DATA"], ".cache")
        cache_dir = "/data-gauss/oscdav/.cache"
        if self.hparams.dataset == "owt":
            if split == "train":
                split = "train[:-100000]"
            else:
                split = "train[-100000:]"
        dataset = datasets.load_dataset(
            "lm1b" if self.hparams.dataset == "lm1b" else "openwebtext",
            streaming=False,
            split=split,
            keep_in_memory=False,
            cache_dir=cache_dir,
        )
        if self.hparams.dataset == "lm1b":
            dataset.set_transform(
                lm1b_detokenizer,
            )
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token)[0]
        self.eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        return _AlignDataset(dataset, self.tokenizer, self.hparams.max_length, shuffle=(split == "train"))

    def setup(self, stage: str):
        self._load_tokenizer()

        self.train_dataset = self._load_dataset("train")
        self.val_dataset = self._load_dataset("test")
        self.test_dataset = self._load_dataset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=False,
        )


if __name__ == "__main__":
    lm = LM1BDataModule()
    lm.setup("fit")
    dl = lm.train_dataloader()
    it = iter(dl)
    samples = []
    for i in range(10):
        example = next(it)["input_ids"]
        print(example.shape)
        import ipdb; ipdb.set_trace()
        samples += [example]
