import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
from osgeo import gdal
import torchvision
from utils import utils
import matplotlib.pyplot as plt

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

JITTER = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1)

class MSDataset(Dataset):

    def __init__(self, dataframe, root_dir, test_flag=False, transform=None):
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
        self.test_flag = test_flag

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
        
        raster = gdal.Open(tile_name)
        tile = np.empty([8,255,255])
        for band in range( raster.RasterCount ):
            tile[band,:,:] = raster.GetRasterBand(band+1).ReadAsArray()
        value = row.wealthpooled.astype('float')          
        tile= torch.from_numpy(np.nan_to_num(tile))

        # We only select MS bands
        tile = tile[:7,:,:]  
        tile = utils.preprocess_raster(tile, TILE_MIN, TILE_MAX, JITTER)
        if self.transform:
            tile = self.transform(tile)
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            return idx, tile, value
        return tile, value
    


class NLDataset(Dataset):

    def __init__(self, dataframe, root_dir, test_flag=False):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.test_flag = test_flag

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
        
        raster = gdal.Open(tile_name)
        tile = np.empty([8,255,255])
        for band in range( raster.RasterCount ):
            tile[band,:,:] = raster.GetRasterBand(band+1).ReadAsArray()
        value = row.wealthpooled.astype('float')          
        tile= torch.from_numpy(np.nan_to_num(tile))

        nl_tile = tile[-1,:,:]  
        if int(row.year) < 2012:
                nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
        else:
            nl_tile = utils.preprocess_viirs_nightlights(nl_tile)
            nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
        nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
        nl_tile = nl_transforms(nl_tile)
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            return idx, tile, value
        return tile, value
    

class MSNLDataset(Dataset):

    def __init__(self, dataframe, root_dir, test_flag=False, transform=None):
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
        self.test_flag = test_flag

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
        
        raster = gdal.Open(tile_name)
        tile = np.empty([8,255,255])
        for band in range( raster.RasterCount ):
            tile[band,:,:] = raster.GetRasterBand(band+1).ReadAsArray()
        value = row.wealthpooled.astype('float')          
        tile= torch.from_numpy(np.nan_to_num(tile))

        # MS bands
        ms_tile = tile[:7,:,:]  
        ms_tile = utils.preprocess_raster(ms_tile, TILE_MIN, TILE_MAX, jitter=None)
        if self.transform:
            ms_tile = self.transform(tile)

        # NL band
        nl_tile = tile[-1,:,:]  
        if int(row.year) < 2012:
                nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
        else:
            nl_tile = utils.preprocess_viirs_nightlights(nl_tile)
            nl_tile = (nl_tile-MIN_OLD) / (MAX_OLD-MIN_OLD)
        nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
        nl_tile = nl_transforms(nl_tile)

        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            return idx, ms_tile, nl_tile, value
        return  ms_tile, nl_tile, value
    

class VITDataset(Dataset):

    def __init__(self, dataframe, root_dir, test_flag=False, transform=None):
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
        self.test_flag = test_flag

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
        
        raster = gdal.Open(tile_name)
        tile = np.empty([8,255,255])
        for band in range( raster.RasterCount ):
            tile[band,:,:] = raster.GetRasterBand(band+1).ReadAsArray()
        value = row.wealthpooled.astype('float')          
        tile= torch.from_numpy(np.nan_to_num(tile))

        # We only select MS bands
        tile = tile[:7,:,:]  
        tile = utils.preprocess_raster(tile, TILE_MIN, TILE_MAX, JITTER)
        if self.transform:
            tile = self.transform(tile)
        # Close Raster (Safety Measure)
        raster = None

        # Adapt shape to Sentinel-2 pre-trained ViT weights 
        vit_tile = utils.landsat_to_sentinel_tile(tile)
        if self.test_flag:
            return idx, vit_tile, value
        return vit_tile, value