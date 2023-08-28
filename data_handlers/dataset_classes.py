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

MEAN_P=[438.3232237817104,
 398.02422630982477,
 681.2318803940371,
 755.4770551826344,
 769.3797750849969,
 785.5918925987272,
 995.751216110191,
 1194.0752506320287,
 1094.7440327783106,
 866.6740475982914,
 606.9098334931566,
 510.7891378258216]
STD_P=[1122.5668846403948,
 897.6616525410869,
 903.6492503396834,
 1058.1928257183752,
 1112.6553764935,
 1267.7919695057851,
 1688.3122659458277,
 1820.2873026662207,
 1438.8972071294047,
 1161.583392709497,
 908.2722861630933,
 918.8520387119656]
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
    


class FCNDataset(Dataset):

    def __init__(self, dataframe, root_dir, pcp_dict, tmp_dict, test_flag=False, transform=None):
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
        self.pcp_dict = pcp_dict
        self.tmp_dict = tmp_dict

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.dataframe.iloc[idx]         
        sequence_tmp = self.build_tmp_from_dict(row)
        sequence_pcp = self.build_pcp_from_dict(row)
        sequence_tmp=torch.from_numpy(sequence_tmp)
        sequence_pcp=torch.from_numpy(sequence_pcp)

        sequence = torch.cat((sequence_tmp,sequence_pcp), dim=1)
        sequence = sequence.swapaxes(0,1)
        value = row.wealthpooled.astype('float')
        if self.test_flag:
            return idx, sequence, value
        return sequence, value


    def build_tmp_from_dict(self, row):
        sequence_tmp=np.zeros((60,3))
        for year in range(row.year-4, row.year+1):
            difference = year -(row.year-4)
            for month in range(12):
                sequence_tmp[month*(difference+1)] = self.tmp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_tmp = (sequence_tmp-297.82736102405767) / 7.546466012481872
        return sequence_tmp
    
    def build_pcp_from_dict(self, row):
        sequence_pcp=np.zeros((60,1))
        for year in range(row.year-4, row.year+1):
            difference = year - (row.year-4)
            for month in range(12):
                sequence_pcp[month*(difference+1)] = self.pcp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_pcp = [(sequence_pcp[i]-MEAN_P[i%12]) / STD_P[i%12] for i in range(len(sequence_pcp)) ]
        return np.array(sequence_pcp)


class MSNLTDataset(Dataset):

    def __init__(self, dataframe, root_dir, pcp_dict, tmp_dict, with_vit=True, test_flag=False, transform=None):
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
        self.pcp_dict = pcp_dict
        self.tmp_dict = tmp_dict

    def __len__(self):
        return len(self.dataframe)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]            

        # 1. RASTERS
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
        # ms_tile = tile[:7,:,:]  
        # ms_tile = utils.preprocess_raster(ms_tile, TILE_MIN, TILE_MAX, jitter=JITTER)
        # if self.transform:
        #     ms_tile = self.transform(ms_tile)
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
        ms_tile = transforms(vit_tile)
        # NL band
        nl_tile = tile[-1,:,:]  
        nl_tile = (nl_tile-min(MIN_VIIRS, torch.min(nl_tile))) / (max(MAX_VIIRS, torch.max(nl_tile))-min(MIN_VIIRS, torch.min(nl_tile)))
        nl_tile = nl_tile[None,:,:]
        nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
        nl_tile = nl_transforms(nl_tile)
        raster = None

        # 2. TIME-SERIES
        sequence_tmp = self.build_tmp_from_dict(row)
        sequence_pcp = self.build_pcp_from_dict(row)
        sequence_tmp=torch.from_numpy(sequence_tmp)
        sequence_pcp=torch.from_numpy(sequence_pcp)

        sequence = torch.cat((sequence_tmp,sequence_pcp), dim=1)
        sequence = sequence.swapaxes(0,1)


        value = row.wealthpooled.astype('float')
        if self.test_flag:
            return idx, ms_tile, nl_tile, sequence, value
        return ms_tile, nl_tile, sequence, value


    def build_tmp_from_dict(self, row):
        sequence_tmp=np.zeros((60,3))
        for year in range(row.year-4, row.year+1):
            difference = year -(row.year-4)
            for month in range(12):
                sequence_tmp[month*(difference+1)] = self.tmp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_tmp = (sequence_tmp-297.82736102405767) / 7.546466012481872
        return sequence_tmp
    
    def build_pcp_from_dict(self, row):
        sequence_pcp=np.zeros((60,1))
        for year in range(row.year-4, row.year+1):
            difference = year - (row.year-4)
            for month in range(12):
                sequence_pcp[month*(difference+1)] = self.pcp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_pcp = [(sequence_pcp[i]-MEAN_P[i%12]) / STD_P[i%12] for i in range(len(sequence_pcp)) ]
        return np.array(sequence_pcp)