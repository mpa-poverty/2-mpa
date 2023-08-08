import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
from osgeo import gdal
import PIL
import torchvision
from utils import utils
import rasterio
import matplotlib.pyplot as plt

TILE_MIN=[-0.2, -0.0641, -0.0866, -0.0308, -0.00245, 0.0, 0.0]
TILE_MAX=[0.9576, 0.9212, 0.97355, 1.2277, 1.48375, 1.57635, 316.9]
MIN_VIIRS=-0.07087274
MAX_VIIRS=3104.1401
MIN_DMSP=0.0
MAX_DMSP=63.0
MIN_PCP = -9999
MAX_PCP = 17169
MIN_TMP=254.64299
MAX_TMP=320.8393

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

    def __init__(self, dataframe, root_dir, transform=None, test_flag=False):
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

        # NL PREPROCESSING
        if int(row.year) < 2012:
                nl_tile = (nl_tile-MIN_DMSP) / (MAX_DMSP-MIN_DMSP)
        else:
            # DMSP-Like VIIRS Transformation 
            # nl_tile = utils.preprocess_viirs_nightlights(nl_tile)
            nl_tile = (nl_tile-min(MIN_VIIRS, torch.min(nl_tile))) / (max(MAX_VIIRS, torch.max(nl_tile))-min(MIN_VIIRS, torch.min(nl_tile)))
        nl_tile = nl_tile[None,:,:]
        nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
        nl_tile = nl_transforms(nl_tile)
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            return idx, nl_tile, value
        return nl_tile, value
    

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
        ms_tile = utils.preprocess_raster(ms_tile, TILE_MIN, TILE_MAX, jitter=JITTER)
        if self.transform:
            ms_tile = self.transform(ms_tile)
        
        # NL band
        nl_tile = tile[-1,:,:]  
        # if int(row.year) < 2012:
        #         nl_tile = (nl_tile-MIN_DMSP) / (MAX_DMSP-MIN_DMSP)
        # else:
            # nl_tile = utils.preprocess_viirs_nightlights(nl_tile)
        nl_tile = (nl_tile-min(MIN_VIIRS, torch.min(nl_tile))) / (max(MAX_VIIRS, torch.max(nl_tile))-min(MIN_VIIRS, torch.min(nl_tile)))
        nl_tile = nl_tile[None,:,:]
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
        # tile= torch.from_numpy(np.nan_to_num(tile))
        # We only select MS bands
        # vit_tile = utils.preprocess_raster(tile, TILE_MIN, TILE_MAX, JITTER)
        vit_tile = tile[:3,:,:]  
        for i in range(3):
            vit_tile[i] =  (vit_tile[i]  - TILE_MIN[i]) / (TILE_MAX[i] - TILE_MIN[i]) * 255
        vit_tile = np.swapaxes(vit_tile, 0, 2)
        vit_tile = PIL.Image.fromarray(np.uint8(vit_tile)).convert('RGB')
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        vit_tile = transforms(vit_tile)
        # Close Raster (Safety Measure)
        raster = None

        # Adapt shape to Sentinel-2 pre-trained ViT weights 
        # vit_tile = utils.landsat_to_sentinel_tile(vit_tile)
        if self.test_flag:
            return idx, vit_tile, value
        return vit_tile, value
    


class LSTMDataset(Dataset):

    def __init__(self, dataframe, root_dir, test_flag=False, transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.root_dir = root_dir
        self.test_flag = test_flag

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[idx]         
        sequence_pcp = torch.from_numpy(np.array(row.precipitation)).reshape((5,3))
        sequence_pcp = (sequence_pcp - MIN_PCP) / (MAX_PCP - MIN_PCP)
        value = row.wealthpooled.astype('float')     
   
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
        ms_tile = utils.preprocess_raster(ms_tile, TILE_MIN, TILE_MAX, jitter=JITTER)
        if self.transform:
            ms_tile = self.transform(ms_tile)
        
        # NL band
        nl_tile = tile[-1,:,:]  
        nl_tile = (nl_tile-min(MIN_VIIRS, torch.min(nl_tile))) / (max(MAX_VIIRS, torch.max(nl_tile))-min(MIN_VIIRS, torch.min(nl_tile)))
        nl_tile = nl_tile[None,:,:]
        nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
        nl_tile = nl_transforms(nl_tile)
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            return idx, sequence_pcp, ms_tile, nl_tile,
        return sequence_pcp, ms_tile, nl_tile, value
