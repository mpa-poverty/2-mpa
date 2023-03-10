import torch
import pandas as pd
from torch.utils.data import Dataset


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, label, columns_to_keep, transform=None):
        self.data = pd.read_csv(csv_path)
        self.label = label
        self.columns_to_keep = columns_to_keep
        self.transform = transform

    def __getitem__(self, index):
        points = self.data[columns_to_keep][index]
        label = self.labels[index]

        return points, label

    def __len__(self):
        return len(self.data)