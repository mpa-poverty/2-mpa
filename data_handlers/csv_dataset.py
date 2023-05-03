import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import rasterio as rio
from utils import utils

class CustomDatasetFromDataFrame(Dataset):

    def __init__(self, dataframe, root_dir, tile_min, tile_max, nl=False, transform=None,):
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
        self.nl = nl
        self.tile_min = tile_min
        self.tile_max = tile_max

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tile_name = os.path.join(self.root_dir, 
                                str(self.dataframe.iloc[idx, -1])
                                )

        tile = np.array(rio.open(tile_name).read())
        tile= torch.from_numpy(np.nan_to_num(tile))
        value = self.dataframe.iloc[idx, -3].astype('float')
        if self.transform:
            tile = self.transform(tile)

        for i in range(8):
            tile[i] = (tile[i]-self.tile_min[i]) / (self.tile_max[i]-self.tile_min[i])
        ms_tile = tile[:7,:,:]
        if self.nl:
            nl_tile = tile[7:,:,:]
            return ms_tile, nl_tile, value
        return ms_tile, value
        