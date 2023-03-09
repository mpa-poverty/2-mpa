from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.constants import Units
from torchgeo.utils import _to_tuple, tile_to_chips
from torch.samplers import single
from csv_utils import read_rows

class PointGeoSampler(GeoSampler):
  """Samples from an intersection of points and image GeoDatasets.
  This sampler should take in a pair of coordinates and region of interests and
  returns image patches containing the coordinates of interest.
  """

  def __init__(
      self,
      dataset: GeoDataset,   # raster dataset
      rows_list: list[int], # pre_sampled csv rows indices
      csv_path: str,
      size: Union[Tuple[float, float], float],
      roi: Optional[BoundingBox] = None,
      units:  Units = Units.PIXELS,
  )-> None:

    super().__init__(dataset, roi)
    self.rows_list = rows_list
    self.dataset = dataset
    self.csv = read_rows(csv_path, self.rows_list)
    . 
    .
    .
    for row in dataset.rows:
        lon/lat = ... 
        minx = x-self.size[0]//2
        maxx = x+self.size[0]//2
        miny = y-self.size[1]//2
        maxy = y+self.size[1]//2
        # warning : time resolution is ambiguous 
        yield BoundingBox(minx, miny, maxx, maxy, row_year.mint, row_year.maxt)

    def __len__(self) -> int:
        return len(self.rows_list)