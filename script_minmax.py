import glob
import rasterio as rio
import torch
from tqdm import tqdm
import os
import numpy as np


def main():
    DATA_DIR = os.path.join('data','landsat_tif','')
    tile_names=glob.glob(os.path.join(DATA_DIR)+'*.tif')
    maxs = torch.tensor([-1000,-1000,-1000,-1000,-1000,-1000,-1000])
    mins = torch.tensor([1000,1000,1000,1000,1000,1000,1000])
    for tile in tqdm(tile_names):
        tiles=torch.from_numpy(np.nan_to_num(np.array(rio.open(tile).read())))
        mins = torch.minimum(mins, tiles.view(7, -1).min())                      
        maxs = torch.maximum(maxs, tiles.view(7, -1).max())
    
    print("mins: ",mins, "\nmaxs: ", maxs)
    return

if __name__ == "__main__":
    main()