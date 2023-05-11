import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

import utils
from tokens_extractor import extract_tokens


class Tables(Dataset):
    def __collate_fn(samples):
        tokens = torch.nn.utils.rnn.pad_sequence(
            list(map(lambda x: x["input_ids"], samples))
        )
        labels = torch.cat(list(map(lambda x: x["labels"], samples)))

        return {"input_ids": tokens.T, "labels": labels}

    def __init__(
        self,
        paths,
        tokenizer,
        labels_encoder,
        use_rand=False,
        max_tokens=200,
        max_columns=20,
        max_tokens_per_column=200,
        sep="|",
    ):
        self.paths = paths
        self.use_rand = use_rand
        self.tokenizer = tokenizer
        self.labels_encoder = labels_encoder
        self.max_tokens = max_tokens
        self.max_columns = max_columns
        self.max_tokens_per_column = max_tokens_per_column
        self.sep = sep
        np.random.seed(42)

    def __len__(self):
        return len(self.paths) - 1

    def __getitem__(self, idx):
        df = pd.read_csv(self.paths[idx], sep=self.sep)

        labels = utils.get_labels(self.paths[idx])

        assert len(labels) == len(df.columns)

        columns = df.columns[: self.max_columns]
        labels = labels[: self.max_columns]

        if self.use_rand:
            cols_order, labels = zip(*np.random.permutation(list(zip(columns, labels))))
            df = df[list(cols_order)]
            df = shuffle(df)

        table_tokens = extract_tokens(
            df,
            self.tokenizer,
            len(labels),
            self.max_tokens,
            self.max_columns,
            self.max_tokens_per_column,
        )

        labels = self.labels_encoder.transform(labels)[: self.max_columns]

        return {"input_ids": torch.tensor(table_tokens), "labels": torch.tensor(labels)}

    def create_dataloader(self, batch_size=40, num_workers=1, shuffle=False):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=Tables.__collate_fn,
        )


class Columns(Dataset):
    def __init__(self, path, tokenizer, labels_encoder):
        self.df = pd.read_csv(path, sep="<")
        self.tokenizer = tokenizer
        self.labels_encoder = labels_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels_encoder.transform([str(self.df["label"].iloc[idx])])[0]
        str_repr_of_column = str(self.df["column"].iloc[idx])
        tokens = self.tokenizer(
            str_repr_of_column, truncation=True, padding="max_length", max_length=30
        ).input_ids

        return {"input_ids": torch.tensor(tokens), "labels": torch.tensor(label)}

    def create_dataloader(self, batch_size=40, num_workers=1, shuffle=False):
        return DataLoader(
            self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )
