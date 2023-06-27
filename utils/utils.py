import torch
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import cv2 as cv
from shapely.geometry import Point, Polygon
from data_handlers import dataset_classes
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
    if countryname in ['cote_d_ivoire',"Côte d'Ivoire","Côte D'Ivoire"]:
        return "Côte d'Ivoire"
    if countryname in ['democratic_republic_of_congo','Democratic Republic of the Congo','Democratic Republic Of The Congo'] :
        return 'Democratic Republic of the Congo'
    if countryname in ['tanzania','Tanzania','United Republic Of Tanzania','United Republic of Tanzania']:
        return 'United Republic of Tanzania'
    return countryname.replace('_', ' ').title()


def preprocess_viirs_nightlights(viirs_tile):
    viirs_tile=viirs_tile.numpy()[0,:,:]
    tile_shape = viirs_tile.shape
    # Resize to match arc.second-1 pixels
    v = skimage.transform.resize(viirs_tile, (16,16))
    # ~ Kernel Density estimation
    vkd = cv.GaussianBlur(v,(11,11),0)
    # Match DMSP resolution
    vkd = skimage.transform.resize(vkd, (8,8))
    # Log transform
    log_vkd = np.log(vkd+1)
    # Sigmoid transform -- values from (Li, 2020)
    sig_vkd = 6.5 + 57.4 * ( 1 / ( 1 + np.exp( - 1.9 * (log_vkd - 10.8) ) ) )
    # Return converted raster to its original size
    dmsp_like = skimage.transform.resize(sig_vkd, tile_shape, preserve_range=True, anti_aliasing=False)
    return torch.reshape(torch.tensor(dmsp_like), (1, tile_shape[0], tile_shape[1]))


def preprocess_raster(raster, mins, maxs, jitter=None):
    for i in range(raster.shape[0]):
        raster[i] = (raster[i]-mins[i]) / (maxs[i]-mins[i])
        # Color Jittering transform
        tmp_shape = raster[i].shape
        if jitter:
            raster[i] = torch.reshape(
                jitter(raster[i][None,:,:]), 
                tmp_shape
            )


def datasets_from_model_type(model_type, data, data_dir, data_config, fold_dict, fold, test_flag=False):
    match model_type:
        case 'ms':
            return (
                dataset_classes.MSDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    transform=data_config['train_transform'],
                    test_flag=test_flag),
                dataset_classes.MSDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    transform=data_config['test_transform'],
                    test_flag=test_flag)
            )    
        case 'nl':
            return (
                dataset_classes.NLDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    transform=data_config['train_transform'],
                    test_flag=test_flag),
                dataset_classes.NLDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    transform=data_config['test_transform'],
                    test_flag=test_flag)
            )      
        case 'msnl':
            return (
                dataset_classes.MSNLDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    transform=data_config['train_transform'],
                    test_flag=test_flag),
                dataset_classes.MSNLDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    transform=data_config['test_transform'],
                    test_flag=test_flag)
            )
        
    return None

def testset_from_model_type(model_type, data, data_dir, data_config, fold_dict, fold, test_flag=True):
    match model_type:
        case 'ms':
            return (
                dataset_classes.MSDataset(
                    data.iloc[fold_dict[fold]['test']], 
                    data_dir, 
                    transform=data_config['test_transform'],
                    test_flag=test_flag)
            )    
        case 'nl':
            return (
                dataset_classes.NLDataset(
                    data.iloc[fold_dict[fold]['test']], 
                    data_dir, 
                    transform=data_config['test_transform'],
                    test_flag=test_flag)
            )      
        case 'msnl':
            return (
                dataset_classes.MSNLDataset(
                    data.iloc[fold_dict[fold]['test']], 
                    data_dir, 
                    transform=data_config['test_transform'],
                    test_flag=test_flag)
            )
        
    return None


def landsat_to_sentinel_tile(tile):
    result = np.zeros((13, tile.shape[1], tile.shape[2]))
    result[0,:,:] = tile[0,:,:]
    result[1,:,:] = tile[1,:,:]
    result[2,:,:] = tile[2,:,:]
    result[3,:,:] = tile[3,:,:]
    result[4,:,:] = tile[3,:,:]
    result[5,:,:] = tile[4,:,:]
    result[6,:,:] = tile[4,:,:]
    result[7,:,:] = tile[4,:,:]
    result[8,:,:] = tile[4,:,:]
    result[9,:,:] = tile[4,:,:]
    result[10,:,:] = tile[4,:,:]
    result[11,:,:] = tile[5,:,:]
    result[12,:,:] = tile[6,:,:]
    return result