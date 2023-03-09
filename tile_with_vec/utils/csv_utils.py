import random
import string
import pandas as pd



def read_rows(csv_path:str, rows:list[int]):

    return pd.read_csv(csv_path, skiprows = lambda x: x not in rows)


def train_test_split_csv(csv_path:str, cutoff:float=0.5, total_rows:int=None) -> pd.DataFrame:
    if cutoff > 1:
        raise ValueError(f"cutoff must be set between 0. and 1.")
    if total_rows==None:
        with open(csv_path,"r") as f:
            total_rows = sum(1 for row in f)

    train_indices = random.sample(range(1,total_rows+1), int(cutoff*total_rows))
    test_indices = [i for i in range(1, total_rows+1) if i not in train_indices]
    
    return train_indices, test_indices












import abc
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.constants import Units
from torchgeo.utils import _to_tuple, tile_to_chips
from torch.samplers import single


class PointGeoSampler(GeoSampler):
  """Samples from an intersection of points and image GeoDatasets.

  This sampler should take in a pair of coordinates and region of interests and
  returns image patches containing the coordinates of interest.
  """

  def __init__(
      self,
      dataset: GeoDataset,  ## for exampler raster dataset of sentinel 2
      points: GeoDataset,   ## e.g GBIF dataset containing coordinate points 
      roi: Optional[BoundingBox] = None,
      shuffle: bool = False,
      size: Union[Tuple[float, float], float],
      units:  Units = Units.PIXELS,
  )-> None:

    super().__init__(dataset, roi)
    self.shuffle = shuffle
    self.points = points
    self.dataset = dataset
    self.hits = []

    # Adjust CRS to the dataset CRS
    self.points.crs = self.dataset.crs
    # Keep the intersection of the two datasets
    for point in point_dataset.index.intersection(point_dataset.index.bounds, objects=True):
        if list(self.index.intersection(point.bounds)):
            self.hits.append(point)

    # The proposed version on the original Pull Request 
    # was returning the cropped dataset covering the whole
    # point cloud. Instead we want to sample points and 
    # return small bounding boxes CENTERED around these:
    def __iter__(self) -> Iterator[BoundingBox]:
    generator: Callable[[int], Iterable[int]] = range
    if self.shuffle: 
        generator = torch.randperm
    for idx in generator(len(self)):
        x, y = self.hits[idx]
        minx = x-self.size[0]//2
        maxx = x+self.size[0]//2
        miny = y-self.size[1]//2
        maxy = y+self.size[1]//2
        yield BoundingBox(minx, miny, maxx, maxy, dataset.bounds.mint, dataset.bounds.maxt)
        
    def __len__(self) -> int:
        return len(self.hits)