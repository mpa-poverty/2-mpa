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
  """Samples a bounding box around a given point in an image GeoDataset.
  """

  def __init__(
      self,
      dataset: GeoDataset,   # raster dataset
      lat: float, # pre_sampled csv rows indices
      lon: float,
      size: Union[Tuple[float, float], float],
      year,
      month=None,
      day=None,
      units:  Units = Units.PIXELS,
  )-> None:
    
    self.dataset = dataset
    self.lat = lat
    self.lon = lon
    self.size = size//2
    self.year = year
    self.month = month
    self.day = day
    # Bounding box coordinates
    minx = self.lat - size
    miny = self.lon - size
    maxx = self.lat + size
    maxy = self.lon + size
    mint, maxt = disambiguate_timestamps(year, month, day)

    yield BoundingBox(minx, miny, maxx, maxy, mint, maxt)

    def __len__(self) -> int:
        return 1