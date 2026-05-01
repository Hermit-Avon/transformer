from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

PAD = "<pad>"
UNK = "<unk>"


class Vocab:
    def __init__(self, tokens, min_freq=2, specials=(PAD, UNK, "<sos>", "<eos>")):
        counter = Counter(tokens)
        self.itos = []
        for token in specials:
            if token not in self.itos:
                self.itos.append(token)
        for word, count in counter.items():
            if count >= min_freq and word not in self.itos:
                self.itos.append(word)
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def lookup(self, word):
        return self.stoi.get(word, self.stoi[UNK])


@dataclass
class FieldLike:
    vocab: Vocab | None = None


@dataclass
class Batch:
    src: torch.Tensor
    trg: torch.Tensor


class ParallelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class DataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenizers = {"en": tokenize_en, "de": tokenize_de}
        self.init_token = init_token
        self.eos_token = eos_token
        self.source = FieldLike()
        self.target = FieldLike()
        self.src_lang = ext[0].lstrip(".")
        self.trg_lang = ext[1].lstrip(".")
        self.data_dir = Path("data/multi30k")
        self._specials = (PAD, UNK, init_token, eos_token)
        print("dataset initializing start")

    def _tokenize(self, text, lang):
        return self.tokenizers[lang](text.lower())

    def _load_pairs(self, split):
        src_file = self.data_dir / f"{split}.{self.src_lang}"
        trg_file = self.data_dir / f"{split}.{self.trg_lang}"
        if not src_file.exists() or not trg_file.exists():
            raise FileNotFoundError(f"Missing dataset files: {src_file} and/or {trg_file}")

        pairs = []
        with src_file.open("r", encoding="utf-8") as src_f, trg_file.open("r", encoding="utf-8") as trg_f:
            for src_text, trg_text in zip(src_f, trg_f):
                src_text = src_text.strip()
                trg_text = trg_text.strip()
                if src_text and trg_text:
                    pairs.append((
                        self._tokenize(src_text, self.src_lang),
                        self._tokenize(trg_text, self.trg_lang),
                    ))
        return pairs

    def make_dataset(self):
        train_data = ParallelDataset(self._load_pairs("train"))
        valid_data = ParallelDataset(self._load_pairs("valid"))
        test_data = ParallelDataset(self._load_pairs("test"))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        src_tokens = []
        trg_tokens = []
        for src, trg in train_data.pairs:
            src_tokens.extend(src)
            trg_tokens.extend(trg)
        self.source.vocab = Vocab(src_tokens, min_freq=min_freq, specials=self._specials)
        self.target.vocab = Vocab(trg_tokens, min_freq=min_freq, specials=self._specials)

    def _numericalize(self, tokens, vocab):
        ids = [vocab.lookup(self.init_token)]
        ids.extend(vocab.lookup(token) for token in tokens)
        ids.append(vocab.lookup(self.eos_token))
        return torch.tensor(ids, dtype=torch.long)

    def _build_collate(self):
        if self.source.vocab is None or self.target.vocab is None:
            raise ValueError("Call build_vocab before make_iter")

        src_vocab = self.source.vocab
        trg_vocab = self.target.vocab
        src_pad_idx = src_vocab.stoi[PAD]
        trg_pad_idx = trg_vocab.stoi[PAD]

        def collate_fn(batch):
            src_batch = [self._numericalize(src, src_vocab) for src, _ in batch]
            trg_batch = [self._numericalize(trg, trg_vocab) for _, trg in batch]
            src_tensor = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
            trg_tensor = pad_sequence(trg_batch, batch_first=True, padding_value=trg_pad_idx)
            return Batch(src=src_tensor, trg=trg_tensor)

        return collate_fn

    def make_iter(self, train, validate, test, batch_size, device):
        base_collate = self._build_collate()

        def collate_fn(batch):
            packed = base_collate(batch)
            return Batch(src=packed.src.to(device), trg=packed.trg.to(device))

        pin_memory = getattr(device, "type", str(device)) != "cpu"
        train_iterator = TorchDataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_memory)
        valid_iterator = TorchDataLoader(validate, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory)
        test_iterator = TorchDataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory)

        print("dataset initializing done")
        return train_iterator, valid_iterator, test_iterator
