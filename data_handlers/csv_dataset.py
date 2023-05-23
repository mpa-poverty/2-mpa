import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
from osgeo import gdal
import torchvision
# from preprocessing.tfrecord.torch.dataset import TFRecordDataset



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
            tile[i] = (tile[i]-self.tile_min[i]) / (self.tile_max[i]-self.tile_min[i])
        ms_tile = tile[:7,:,:]
        if self.transform:
            ms_tile = self.transform(ms_tile)
        if self.nl:
            nl_tile = tile[7:,:,:]
            nl_transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=224)
            )
            nl_tile = nl_transforms(nl_tile)
            return ms_tile, nl_tile, value
        return ms_tile, value
        