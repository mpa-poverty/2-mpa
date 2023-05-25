import torch
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
# 

def configure_optimizer( config, model ):
    if config['optimizer'] in ("Adam", "adam"):
        return torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] in ("SGD", 'sgd'):
        return torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise KeyError(config['optimizer'])


def configure_loss( config ):
    if config['loss'] in ("mse", "l2"):
         return torch.nn.MSELoss()
    elif config['loss'] in ("mae", "l1"):
         return torch.nn.L1Loss()
    else:
        raise KeyError(config['loss'])
    return

# OBSOLETE
# MAX_VALUE = 2.643941
# MIN_VALUE = -1.3713919
# def normalize_asset(asset, min_asset=MIN_VALUE, max_asset=MAX_VALUE):
#     return (asset- min_asset) / (max_asset - min_asset)

# def denormalize_asset(asset, min_asset=MIN_VALUE, max_asset=MAX_VALUE):
#     return asset * (max_asset - min_asset) + min_asset


def compute_average_crossval_results(results:dict):
    result_list=[]
    for fold in results:
        fold_result = [np.array(results[fold]['test_r2'][i].cpu().numpy())[()] for i in range(len(results[fold]['test_r2']))]
        result_list.append(fold_result)
    result_list = np.mean( np.array( result_list ), axis=0 )
    return result_list

    
def convert_csv_to_epsg(csv_name, from_epsg="EPSG:4326", to_epsg="EPSG:3857"):
    # creating a geometry column 
    dataset = pd.read_csv(csv_name)
    geometry = [Point(xy) for xy in zip(dataset['lon'], dataset['lat'])]
    crs = {'init': from_epsg}
    # Creating a Geographic data frame 
    gdf = gpd.GeoDataFrame(dataset, crs=crs, geometry=geometry)
    gdf = gdf.to_crs(to_epsg)
    return gdf

def add_bounding_box_from_geometry(row, resolution=30, extent=127):
    offset = extent * resolution
    p= row.geometry
    xmin, ymin, xmax, ymax = p.x - offset, p.y - offset, p.x + offset, p.y + offset
    bounding_box = Polygon([
        (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, xmin), (xmin, ymin)
    ])
    return bounding_box


def make_config_picklefile(
        path:str,
        csv_path:str,
        fold_path:str,
        mean, 
        std, 
        max_, 
        min_,
        train_transform,
        test_transform
    ):
    config_file=dict()
    config_file['csv']=csv_path
    config_file['fold']=fold_path
    config_file['mean']=mean
    config_file['std']=std
    config_file['max']=max_
    config_file['min']=min_
    config_file['train_transform']=train_transform
    config_file['test_transform']=test_transform
    with open(path, 'wb') as f:
        pickle.dump(config_file, f, protocol=pickle.HIGHEST_PROTOCOL)
    return config_file

def standardize_countryname(countryname:str)->str:
    if countryname=='cote_d_ivoire':
        return "Côte d'Ivoire"
    if countryname=='democratic_republic_of_congo':
        return 'Democratic Republic of the Congo'
    return countryname.replace('_', ' ').title()