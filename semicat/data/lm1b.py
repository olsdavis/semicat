"""
LM1B/OpenWebText DataModule.
"""

from dotenv import load_dotenv
import itertools
import os
import re
from lightning import LightningDataModule
import datasets
import tokenizers
import transformers
import torch
from torch.utils.data import DataLoader


# from .env, data cache dir
_DATASET_CACHE_DIR_KEY = "DATASET_CACHE_DIR"


# Pre-compile regexes
_REGEX_APOSTROPHE = re.compile(r" \'(\w+)")
_REGEX_PERIOD = re.compile(r" (\w+) \. ")
_REGEX_PERIOD_END = re.compile(r" (\w+) \.$")
_REGEX_QUESTION_END = re.compile(r" \?$")
_REGEX_EXCLAMATION_END = re.compile(r" \!$")
_REGEX_QUOTES = re.compile(r"\" ([^\"]+) \"")
_REGEX_SINGLE_QUOTES = re.compile(r"\' ([^\']+) \'")
_REGEX_PARENS = re.compile(r"\( ([^\(\)]+) \)")
_REGEX_BRACKETS = re.compile(r"\[ ([^\[\]]+) \]")


def lm1b_detokenizer(x: str) -> str:
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = _REGEX_APOSTROPHE.sub(r"'\1", x)
    x = _REGEX_PERIOD.sub(r" \1. ", x)
    x = _REGEX_PERIOD_END.sub(r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = _REGEX_QUESTION_END.sub("?", x)
    x = x.replace(" ! ", "! ")
    x = _REGEX_EXCLAMATION_END.sub("!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = _REGEX_QUOTES.sub(r'"\1"', x)
    x = _REGEX_SINGLE_QUOTES.sub(r"'\1'", x)
    x = _REGEX_PARENS.sub(r"(\1)", x)
    x = _REGEX_BRACKETS.sub(r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


class LM1BDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        max_length: int = 1024,
        dataset: str = "lm1b",
    ):
        assert dataset in ["lm1b", "owt"], f"unsupported dataset '{dataset}'"
        super().__init__()
        load_dotenv()
        self.save_hyperparameters(logger=False)
        self.tokenizer = None

    def _load_tokenizer(self):
        if self.hparams.dataset == "lm1b":
            self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        if (isinstance(self.tokenizer, transformers.GPT2TokenizerFast)
            or isinstance(self.tokenizer, transformers.GPT2Tokenizer)):
                self.tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
                    (self.tokenizer.bos_token, self.tokenizer.bos_token_id), (self.tokenizer.eos_token, self.tokenizer.eos_token_id)
                )

        # For wrapped batches:
        #  [BOS] sent1 [EOS] sent2-fragment [EOS]
        #  [BOS] sent2-fragment [EOS] sent3 [EOS]
        if self.tokenizer.bos_token is None:
            if self.tokenizer.cls_token is None:
                raise AttributeError(f'Tokenizer must have a bos_token or cls_token: {self.tokenizer}')
            self.tokenizer.bos_token = self.tokenizer.cls_token
        if self.tokenizer.eos_token is None:
            if self.tokenizer.sep_token is None:
                raise AttributeError(f'Tokenizer must have a eos_token or sep_token: {self.tokenizer}')
            self.tokenizer.eos_token = self.tokenizer.sep_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def _load_dataset(self, split: str):
        assert self.tokenizer is not None, "need tokenizer"

        cache_dir = os.getenv(_DATASET_CACHE_DIR_KEY)
        assert cache_dir is not None, "must specify a cache dir for the data"

        end_file = os.path.join(cache_dir, f"{self.hparams.dataset}_{self.hparams.max_length}_{split}_processed")
        if os.path.exists(end_file):
            print(f"Loading processed dataset from {end_file}")
            return datasets.load_from_disk(end_file).with_format("torch")

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
            trust_remote_code=True,
        )

        self.eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        self.bos = self.tokenizer.encode(self.tokenizer.bos_token)[0]

        # pre-process the dataset
        def preprocess(example):
            text = example["text"]

            if self.hparams.dataset == "lm1b":
                text = [lm1b_detokenizer(t) for t in text]

            tokens = self.tokenizer(text, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
            tokens = {"input_ids": [t + [self.eos] for t in tokens["input_ids"]]}
            return tokens

        # take a subset of the dataset
        dataset = dataset.map(preprocess, batched=True, num_proc=32, load_from_cache_file=True, desc="Tokenizing", batch_size=1024)
        dataset = dataset.remove_columns("text")

        # now, group blocks
        def group_texts(examples):
            concatenated_examples = list(itertools.chain(*examples["input_ids"]))
            length = len(concatenated_examples)
            new_block_size = self.hparams.max_length - 2  # [BOS] and [EOS] to be added
            total_length = (length // new_block_size) * new_block_size
            # Split by chunks of max_len.
            _values = []
            for i in range(0, total_length, new_block_size):
                _values.append([self.bos] + concatenated_examples[i : i + new_block_size] + [self.eos])
            return {"input_ids": _values}

        dataset = dataset.map(group_texts, batched=True, num_proc=32, load_from_cache_file=True, desc="Grouping texts")
        print(f"Saving processed dataset to {end_file}")
        dataset.save_to_disk(end_file)
        dataset = dataset.with_format("torch")
        return dataset

    def setup(self, stage: str):
        self._load_tokenizer()

        self.train_dataset = self._load_dataset("train")
        self.val_dataset = self._load_dataset("test")
        self.test_dataset = self._load_dataset("test")

    def tensor_to_strings(self, batch: torch.Tensor) -> list[str]:
        assert self.tokenizer is not None, "need tokenizer"
        assert batch.shape == (batch.size(0), self.hparams.max_length)
        ret = self.tokenizer.batch_decode(batch)
        return ret

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
    lm = LM1BDataModule(max_length=128, dataset="lm1b", batch_size=1024)
    lm.setup("fit")
    dl = lm.train_dataloader()
    it = iter(dl)
    for i in range(5):
        example = next(it)
        print(example["input_ids"].shape, example["input_ids"].max())
