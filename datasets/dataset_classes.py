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

NORMALIZER = 'normalizer.pkl'
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

    def __init__(self, dataframe, root_dir, normalizer, test_flag=False):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.normalizer = normalizer
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
        transforms=torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(size=224),
            torchvision.transforms.RandomVerticalFlip(size=224)
        )
        tile = transforms(tile)
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=None)
            return idx, tile, value
        tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], JITTER)
        return tile, value
    


class NLDataset(Dataset):

    def __init__(self, dataframe, root_dir, normalizer, transform=None, test_flag=False):
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
        self.normalizer= normalizer

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
        
        transforms=torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(size=224),
            torchvision.transforms.RandomVerticalFlip(size=224)
        )
        nl_tile = transforms(nl_tile)
        nl_tile = (nl_tile-self.normalizer['landsat_+_nightlights'][0][-1]) / self.normalizer['landsat_+_nightlights'][1][-1]
        nl_tile = nl_tile[None,:,:]
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            return idx, nl_tile, value
        return nl_tile, value
    

class MSNLDataset(Dataset):

    def __init__(self, dataframe, root_dir, normalizer, test_flag=False, transform=None):
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
        self.normalizer = normalizer
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
        
        # NL band
        nl_tile = tile[-1,:,:]  
       
        transforms=torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(size=224),
            torchvision.transforms.RandomVerticalFlip(size=224)
        )
        nl_tile = transforms(nl_tile)
        nl_tile = (nl_tile-self.normalizer['landsat_+_nightlights'][0][-1]) / self.normalizer['landsat_+_nightlights'][1][-1]
        nl_tile = nl_tile[None,:,:]
        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            ms_tile = tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=None)
            return idx, ms_tile, nl_tile, value
        ms_tile = tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], JITTER)
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

        transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomHorizontalFlip(size=256),
            torchvision.transforms.RandomVerticalFlip(size=256)
        )

        vit_tile = tile[:3,:,:]  
        vit_tile = transforms(vit_tile)
        # Close Raster (Safety Measure)
        raster = None

        # Adapt shape to Sentinel-2 pre-trained ViT weights 
        # vit_tile = utils.landsat_to_sentinel_tile(vit_tile)
        if self.test_flag:
            vit_tile = utils.preprocess_landsat(vit_tile, self.normalizer, jitter=None)
            return idx, vit_tile, value
        vit_tile = utils.preprocess_landsat(vit_tile, self.normalizer, JITTER)    
        return vit_tile, value
    


class FCNDataset(Dataset):

    def __init__(self, dataframe, root_dir, pcp_dict, tmp_dict, normalizer, test_flag=False):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.normalizer = normalizer
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
        sequence_tmp = (sequence_tmp-self.normalizer['temperature'][0]) / self.normalizer['temperature'][1]
        return sequence_tmp
    
    def build_pcp_from_dict(self, row):
        sequence_pcp=np.zeros((60,1))
        for year in range(row.year-4, row.year+1):
            difference = year - (row.year-4)
            for month in range(12):
                sequence_pcp[month*(difference+1)] = self.pcp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_pcp = [(sequence_pcp[i]-self.normalizer['temperature'][0][i%12]) / self.normalizer['temperature'][1][i%12] for i in range(len(sequence_pcp)) ]
        return np.array(sequence_pcp)


class MSNLTDataset(Dataset):

    def __init__(self, dataframe, root_dir, pcp_dict, tmp_dict, normalizer, with_vit=True, test_flag=False):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.normalizer = normalizer
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
        
        # NL band
        nl_tile = tile[-1,:,:]  
       
        transforms=torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(size=224),
            torchvision.transforms.RandomVerticalFlip(size=224)
        )
        nl_tile = transforms(nl_tile)
        nl_tile = (nl_tile-self.normalizer['landsat_+_nightlights'][0][-1]) / self.normalizer['landsat_+_nightlights'][1][-1]
        nl_tile = nl_tile[None,:,:]
        raster=None

        
        # 2. TIME-SERIES
        sequence_tmp = self.build_tmp_from_dict(row)
        sequence_pcp = self.build_pcp_from_dict(row)
        sequence_tmp=torch.from_numpy(sequence_tmp)
        sequence_pcp=torch.from_numpy(sequence_pcp)

        sequence = torch.cat((sequence_tmp,sequence_pcp), dim=1)
        sequence = sequence.swapaxes(0,1)


        value = row.wealthpooled.astype('float')
        if self.test_flag:
            ms_tile = tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=None)
            return idx, ms_tile, nl_tile, sequence, value
        ms_tile = tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=JITTER)
        return ms_tile, nl_tile, sequence, value


    def build_tmp_from_dict(self, row):
        sequence_tmp=np.zeros((60,3))
        for year in range(row.year-4, row.year+1):
            difference = year -(row.year-4)
            for month in range(12):
                sequence_tmp[month*(difference+1)] = self.tmp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_tmp = (sequence_tmp-self.normalizer['temperature'][0]) / self.normalizer['temperature'][1]
        return sequence_tmp
    
    def build_pcp_from_dict(self, row):
        sequence_pcp=np.zeros((60,1))
        for year in range(row.year-4, row.year+1):
            difference = year - (row.year-4)
            for month in range(12):
                sequence_pcp[month*(difference+1)] = self.pcp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_pcp = [(sequence_pcp[i]-self.normalizer['temperature'][0][i%12]) / self.normalizer['temperature'][1][i%12] for i in range(len(sequence_pcp)) ]
        return np.array(sequence_pcp)
    

class VIT_MSNLTDataset(Dataset):

    def __init__(self, dataframe, root_dir, pcp_dict, tmp_dict, normalizer, with_vit=True, test_flag=False):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.normalizer = normalizer
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
        transforms_vit=torch.nn.Sequential(
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomHorizontalFlip(size=256),
            torchvision.transforms.RandomVerticalFlip(size=256)
        )

        ms_tile = tile[:3,:,:]  
        ms_tile = transforms_vit(ms_tile)
        # Close Raster (Safety Measure)
        raster = None

        # NL band
        nl_tile = tile[-1,:,:]  
       
        transforms=torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(size=224),
            torchvision.transforms.RandomVerticalFlip(size=224)
        )
        nl_tile = transforms(nl_tile)
        nl_tile = (nl_tile-self.normalizer['landsat_+_nightlights'][0][-1]) / self.normalizer['landsat_+_nightlights'][1][-1]
        nl_tile = nl_tile[None,:,:]
        raster=None

        
        # 2. TIME-SERIES
        sequence_tmp = self.build_tmp_from_dict(row)
        sequence_pcp = self.build_pcp_from_dict(row)
        sequence_tmp=torch.from_numpy(sequence_tmp)
        sequence_pcp=torch.from_numpy(sequence_pcp)

        sequence = torch.cat((sequence_tmp,sequence_pcp), dim=1)
        sequence = sequence.swapaxes(0,1)


        value = row.wealthpooled.astype('float')
        if self.test_flag:
            ms_tile = tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=None)
            return idx, ms_tile, nl_tile, sequence, value
        ms_tile = tile = utils.preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=JITTER)
        return ms_tile, nl_tile, sequence, value


    def build_tmp_from_dict(self, row):
        sequence_tmp=np.zeros((60,3))
        for year in range(row.year-4, row.year+1):
            difference = year -(row.year-4)
            for month in range(12):
                sequence_tmp[month*(difference+1)] = self.tmp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_tmp = (sequence_tmp-self.normalizer['temperature'][0]) / self.normalizer['temperature'][1]
        return sequence_tmp
    
    def build_pcp_from_dict(self, row):
        sequence_pcp=np.zeros((60,1))
        for year in range(row.year-4, row.year+1):
            difference = year - (row.year-4)
            for month in range(12):
                sequence_pcp[month*(difference+1)] = self.pcp_dict[ (row.country, row.year, year, int(row.cluster)) ][month]   
        sequence_pcp = [(sequence_pcp[i]-self.normalizer['temperature'][0][i%12]) / self.normalizer['temperature'][1][i%12] for i in range(len(sequence_pcp)) ]
        return np.array(sequence_pcp)