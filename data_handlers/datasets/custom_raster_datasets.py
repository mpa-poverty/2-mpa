import os
import torch
import torchgeo
# from torch.utils.data import DataLoader
# from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
# from torchgeo.datasets.utils import download_url
# from torchgeo.samplers import RandomGeoSampler

# DUMMY CLASS
class DummyCustomRasterDataset(RasterDataset):
    # regex filename expression (unique per scene) 
    filename_glob = "T*_B02_10m.tif"    
    # if you have time stamps in the filenames                              
    filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])"
    # date format
    date_format = "%Y%m%dT%H%M%S"
    # set to False if the dataset features masks
    is_image = True
    # True if bands come in separate files
    separate_files = True
    # [Optional] band description
    all_bands = ["B02", "B03", "B04", "B08"]
    rgb_bands = ["B04", "B03", "B02"]



