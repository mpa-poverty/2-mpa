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

    def open_as_array(self, idx, invert=False, rgb_only=True):
        file = np.array(rio.open(os.path.join(self.root_dir,str(idx)+'.tif')).read())
        file_rgb = np.nan_to_num(file)
        file_rgb = file_rgb[:3,:,:]
        file_rgb = file_rgb.transpose((2,1,0))
        file_rgb = file_rgb / file_rgb.max()
        plt.figure()
        plt.imshow((file_rgb*255).astype(int))

        return
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tile_name = os.path.join(self.root_dir,
                                str(idx)+'.tif')
        tile = skimage.io.imread(tile_name)
        value = self.csv.iloc[idx, -3].astype('float')
        sample = {'tile': tile, 'value': value}

        if self.transform:
            sample = self.transform(sample)

        return sample
