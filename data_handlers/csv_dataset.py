import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
import rasterio as rio
from utils import utils

class CustomDatasetFromDataFrame(Dataset):

    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tile_name = os.path.join(self.root_dir,
                                str(self.dataframe.iloc[idx, -1])
                                )
        # tile = skimage.io.imread(tile_name)
        tile = np.array(rio.open(tile_name).read())
        tile= torch.from_numpy(np.nan_to_num(tile))
        value = self.dataframe.iloc[idx, -2].astype('float')
        if self.transform:
            tile = self.transform(tile)

        # Normalize tile
        tile_max = tile.max(dim=0).values
        tile_normed = tile / tile_max
        value = utils.normalize_asset(value)

        return tile_normed, value
    
    def set_transform(self, transform):
        self.transform = transform
        return