'''
Custom GeoDataset for Torchgeo pipeline 
converted from a csv file with point (lat/lon) data.
'''

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox

from utils import utils

class GeoVectorDataset(GeoDataset):
    '''
    GeoDataset that turns a CSV with geopoint data into a torchgeo GeoDataset.
    '''
    res = 0
    _crs = CRS.from_epsg(4326)
    
    def __init__(self, columns:list[str], root:str = 'data')->None:
        super().__init__()
        self.root = root
        files = glob.glob(os.path.join( root, "**.csv" ))
        if not files:
            raise FileNotFoundError(f"Dataset not found in `root={self.root}`")

        if len(columns) < 5:
            raise ValueError(f"The dataset should contain at least ['lat', 'lon', 'year', 'month', 'day'].")
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        data = pd.read_table(
            files[0],
            engine="c",
            usecols=columns,
        )

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for col_variables in data.itertuples(index=False, name=None):
            # Skip rows without lat/lon
            if np.isnan(col_variables[1]) or np.isnan(col_variables[0]):
                continue
            mint, maxt = utils.disambiguate_timestamps(col_variables[2], col_variables[3], col_variables[4])

            coords = (col_variables[0], col_variables[0], col_variables[1], col_variables[1], mint, maxt)
            self.index.insert(i, coords)
            i += 1


        def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """@torchgeo
        Retrieve metadata indexed by query.
        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        bboxes = [hit.bbox for hit in hits]

        if not bboxes:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        sample = {"crs": self.crs, "bbox": bboxes}
        
        return sample

