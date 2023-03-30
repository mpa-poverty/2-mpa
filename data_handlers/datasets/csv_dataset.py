import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from pathlib import Path
import rasterio as rio


class CustomDatasetFromCSV(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tile_name = os.path.join(self.root_dir,
                                str(self.csv.iloc[idx, -1])
                                )
        # tile = skimage.io.imread(tile_name)
        tile = np.array(rio.open(tile_name).read())
        tile= torch.from_numpy(np.nan_to_num(tile))
        value = self.csv.iloc[idx, -3].astype('float')
        sample = {'tile': tile, 'value': value}

        if self.transform:
            sample['tile'] = self.transform(sample['tile'])

        return sample
    
    def set_transform(self, transform):
        self.transform = transform
        return