import torch
import numpy as np
from scipy.special import logit

from torch.utils.data import Dataset, DataLoader


def collate_examples(batch):
    texts, labels = list(zip(*batch))
    max_len = max(len(x) for x in texts)
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for idx, text in enumerate(texts):
        tokens[idx, :len(text)] = np.array(text)
    token_tensor = torch.from_numpy(tokens)

    if all([label is None for label in labels]):
        return (token_tensor,)
    else:
        return token_tensor, torch.FloatTensor(labels)


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, truncate_len, apply_logit, shuffle):
        if apply_logit:
            labels = logit(labels)
        self.target = labels
        self.truncate_len = truncate_len
        self.shuffle = shuffle
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(
            self.texts[idx], max_length=self.truncate_len, truncation=True)

        if self.target is not None:
            return tokens, self.target[idx]

        return tokens, None

    def loader(self, batch_size):
        return DataLoader(
            self,
            collate_fn=collate_examples,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=self.shuffle
        )
