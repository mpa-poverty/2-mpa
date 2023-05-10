import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import rasterio as rio
from utils import utils
from tfrecord.torch.dataset import TFRecordDataset



BANDS            = ['BLUE','GREEN','RED','NIR','SWIR1','SWIR2','TEMP1','NIGHTLIGHTS']
DESCRIPTOR       = {
                'cluster':"float",
                'lat':"float", 
                "lon":"float",
                'wealthpooled':"float",
                'BLUE':"float",
                'GREEN':"float",
                'RED':"float",
                'NIR':"float",
                'SWIR1':"float",
                'SWIR2':"float",
                'TEMP1':"float",
                'NIGHTLIGHTS':"float"
              }   


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
        row = self.dataframe.iloc[idx]
        tile_name = os.path.join(self.root_dir,
                                 str(row.country)+"_"+str(row.year),
                                 str(row.cluster)+".tfrecord"
                                 )                 
                                
        value = row.wealthpooled.astype('float')
        dataset = TFRecordDataset(tile_name, 
                                 index_path=None, 
                                 description=DESCRIPTOR)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        iterator = iter(loader)
        tiles = []
        tile = None
        while (data := next(iterator, None)) is not None:
            for i in range(len(BANDS)):
                new_arr = data[BANDS[i]][0].numpy().reshape((255,255))
                tiles.append(new_arr)
            tile = np.swapaxes(np.array(tiles), 0, 2 )
        
        tile= torch.from_numpy(np.nan_to_num(tile))
        

        for i in range(8):
            tile[i] = (tile[i]-self.tile_min[i]) / (self.tile_max[i]-self.tile_min[i])
        ms_tile = tile[:7,:,:]
        if self.transform:
            ms_tile = self.transform(ms_tile)
        if self.nl:
            nl_tile = tile[7:,:,:]
            return ms_tile, nl_tile, value
        
        return ms_tile, value
        