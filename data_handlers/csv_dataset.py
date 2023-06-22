import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
from osgeo import gdal
import torchvision
from utils import utils
import matplotlib.pyplot as plt
# from preprocessing.tfrecord.torch.dataset import TFRecordDataset

TILE_MIN=[-0.0994, -0.0574, -0.0318, -0.0209, -0.0102, -0.0152, 0.0]
TILE_MAX=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 316.7]
MIN_OLD=0.0
MAX_OLD=63.0
MIN_NEW=-0.07087274
MAX_NEW=3104.1401

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

    def __init__(self, dataframe, root_dir, dual=False, transform=None,):
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
        self.dual = dual

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[idx]            
        tile_name = os.path.join(self.root_dir,
                                 str(row.country)+"_"+str(row.year),
                                 str(row.cluster)+".tif"
                                 )     
        
        
        import numpy as np
        raster = gdal.Open(tile_name)
        tile = np.empty([8,255,255])
        for band in range( raster.RasterCount ):
            tile[band,:,:] = raster.GetRasterBand(band+1).ReadAsArray()          
        value = row.wealthpooled.astype('float')
        tile= torch.from_numpy(np.nan_to_num(tile))     

        for i in range(7):
            tile[i] = (tile[i]-TILE_MIN[i]) / (TILE_MAX[i]-TILE_MIN[i])
        ms_tile = tile[:7,:,:]
        if self.transform:
            ms_tile = self.transform(ms_tile)
        # Close Raster (Safety Mesure)
        raster = None
        if self.dual:
            nl_tile = tile[7:,:,:]
            # Normalization Provider-Dependent
            if int(row.year) < 2012:
                nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
            else:
                nl_tile = utils.preprocess_viirs_nightlights(nl_tile)
                nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
            nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
            nl_tile = nl_transforms(nl_tile)
            return ms_tile, nl_tile, value
        return ms_tile, value
        

class CustomTestDatasetFromDataFrame(Dataset):

    def __init__(self, dataframe, root_dir, dual=False, transform=None,):
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
        self.dual = dual

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[idx]            
        tile_name = os.path.join(self.root_dir,
                                 str(row.country)+"_"+str(row.year),
                                 str(row.cluster)+".tif"
                                 )     
        
        
        import numpy as np
        raster = gdal.Open(tile_name)
        tile = np.empty([8,255,255])
        for band in range( raster.RasterCount ):
            tile[band,:,:] = raster.GetRasterBand(band+1).ReadAsArray()          
        value = row.wealthpooled.astype('float')
        tile= torch.from_numpy(np.nan_to_num(tile))     

        for i in range(8):
            tile[i] = (tile[i]-TILE_MIN[i]) / (TILE_MAX[i]-TILE_MIN[i])
        ms_tile = tile[:7,:,:]
        if self.transform:
            ms_tile = self.transform(ms_tile)
        # Close Raster (Safety Mesure)
        raster = None
        if self.dual:
            nl_tile = tile[7:,:,:]
            # Normalization Provider-Dependent
            if int(row.year) < 2012:
                nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
            else:
                nl_tile = utils.preprocess_viirs_nightlights(nl_tile)
                nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
            nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
            nl_tile = nl_transforms(nl_tile)   
            return ms_tile, nl_tile, value
        return idx, ms_tile, value