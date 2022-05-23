from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class BERTDataset(Dataset):
    def __init__(self, config):
        dataset = pd.read_csv(config["datapath"], sep='\t', names=["comment", "label"], header=None)
        self.comment = [config["tokenizer"](i, add_special_tokens=True, max_length=config["max_len"],
                                            padding='max_length', return_tensors="pt", truncation=True)
                        for i in dataset["comment"]]
        self.labels = [np.int32(i) for i in dataset["label"]]

    def __getitem__(self, i):
        return self.comment[i], self.labels[i]

    def __len__(self):
        return len(self.labels)
