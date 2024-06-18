# UTILS/UTILS.PY
#
# DESCRIPTION: Contains miscellaneous functions
#
# @MDC, 2023


# IMPORT 
import torch
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import cv2 as cv
from shapely.geometry import Point, Polygon
from datasets import dataset_classes


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
    viirs_tile=viirs_tile.numpy()
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
    return torch.reshape(torch.tensor(dmsp_like), (tile_shape[0], tile_shape[1]))


def preprocess_landsat(raster, normalizer, jitter=None):
    for i in range(7):
        raster[i] = (raster[i]- normalizer[0][i]) / (normalizer[1][i])
        # Color Jittering transform
        tmp_shape = raster[i].shape
        if jitter:
            raster[i] = torch.reshape(
                jitter(raster[i][None,:,:]), 
                tmp_shape
            )    
    return raster


def datasets_from_model_type(model_type, data, data_dir, fold_dict, fold, test_flag=False):
    match model_type:
        case 'ms':
            return (
                dataset_classes.MSDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    test_flag=test_flag),
                dataset_classes.MSDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    test_flag=test_flag)
            )    
        case 'nl':
            return (
                dataset_classes.NLDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    test_flag=test_flag),
                dataset_classes.NLDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    test_flag=test_flag)
            )      
        case 'msnl':
            return (
                dataset_classes.MSNLDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    test_flag=test_flag),
                dataset_classes.MSNLDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    test_flag=test_flag)
            )   
        case 'vit':
            return (
                dataset_classes.VITDataset(
                    data.iloc[fold_dict[fold]['train']], 
                    data_dir, 
                    test_flag=test_flag),
                dataset_classes.VITDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    test_flag=test_flag)
            )   
        case 'ts':
            with open('data/additional_data/temperatures.pickle', 'rb') as f:
                tmp_dict = pickle.load(f)
            with open('data/additional_data/precipitations.pickle', 'rb') as f:
                pcp_dict = pickle.load(f)
            # Add dictionary for time-series
            with open('data/additional_data/conflict_noneg.pickle', 'rb') as f:
                conf_dict = pickle.load(f)
            return (
                dataset_classes.FCNDataset(
                    data.iloc[fold_dict[fold]['train']],
                    data_dir, 
                    pcp_dict=pcp_dict,
                    tmp_dict=tmp_dict,
                    conf_dict=conf_dict,
                    test_flag=test_flag),
                dataset_classes.FCNDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    pcp_dict=pcp_dict,
                    tmp_dict=tmp_dict,
                    conf_dict=conf_dict,
                    test_flag=test_flag)
            )   
        case "msnlt":
            with open('data/additional_data/temperatures.pickle', 'rb') as f:
                tmp_dict = pickle.load(f)
            with open('data/additional_data/precipitations.pickle', 'rb') as f:
                pcp_dict = pickle.load(f)
                # Add dictionary for time-series
            with open('data/additional_data/conflict_noneg.pickle', 'rb') as f:
                conf_dict = pickle.load(f)
            return (
                dataset_classes.MSNLTDataset(
                    data.iloc[fold_dict[fold]['train']],
                    data_dir, 
                    pcp_dict=pcp_dict,
                    tmp_dict=tmp_dict,
                    conf_dict=conf_dict,
                    test_flag=test_flag),
                dataset_classes.MSNLTDataset(
                    data.iloc[fold_dict[fold]['val']], 
                    data_dir, 
                    pcp_dict=pcp_dict,
                    tmp_dict=tmp_dict,
                    conf_dict=conf_dict,
                    test_flag=test_flag)
            )   
    return None


def testset_from_model_type(model_type, data, data_dir, fold_dict, fold, test_flag=True):
    match model_type:
        case 'ms':
            return (
                dataset_classes.MSDataset(
                    data.iloc[fold_dict[fold]['train']],
                    data_dir, 
                    test_flag=test_flag)
            )    
        case 'nl':
            return (
                dataset_classes.NLDataset(
                    data.iloc[fold_dict[fold]['test']], 
                    data_dir, 
                    test_flag=test_flag)
            )      
        case 'msnl':
            return (
                dataset_classes.MSNLDataset(
                    data.iloc[fold_dict[fold]['test']], 
                    data_dir, 
                    test_flag=test_flag)
            )
        case 'vit':
            return (
                dataset_classes.VITDataset(
                    data.iloc[fold_dict[fold]['test']], 
                    data_dir, 
                    test_flag=test_flag)
            )
        case 'fcn':
            with open('data/additional_data/temperatures.pickle', 'rb') as f:
                tmp_dict = pickle.load(f)
            with open('data/additional_data/precipitations.pickle', 'rb') as f:
                pcp_dict = pickle.load(f)
                # Add dictionary for time-series
            with open('data/additional_data/conflict_noneg.pickle', 'rb') as f:
                conf_dict = pickle.load(f)
            return (
                dataset_classes.FCNDataset(
                    data.iloc[fold_dict[fold]['test']],
                    data_dir, 
                    pcp_dict=pcp_dict,
                    tmp_dict=tmp_dict,
                    conf_dict=conf_dict,
                    test_flag=test_flag)
            )
        case 'msnlt':
            with open('data/additional_data/temperatures.pickle', 'rb') as f:
                tmp_dict = pickle.load(f)
            with open('data/additional_data/precipitations.pickle', 'rb') as f:
                pcp_dict = pickle.load(f)
                # Add dictionary for time-series
            with open('data/additional_data/conflict_noneg.pickle', 'rb') as f:
                conf_dict = pickle.load(f)
            return ( 
                dataset_classes.MSNLTDataset(
                        data.iloc[fold_dict[fold]['test']], 
                        data_dir, 
                        pcp_dict=pcp_dict,
                        tmp_dict=tmp_dict,
                        conf_dict=conf_dict,
                        test_flag=test_flag)
        
            )   
        case 'ts':
            with open('data/additional_data/temperatures.pickle', 'rb') as f:
                tmp_dict = pickle.load(f)
            with open('data/additional_data/precipitations.pickle', 'rb') as f:
                pcp_dict = pickle.load(f)
                # Add dictionary for time-series
            with open('data/additional_data/conflict_noneg.pickle', 'rb') as f:
                conf_dict = pickle.load(f)
            return ( 
                dataset_classes.FCNDataset(
                        data.iloc[fold_dict[fold]['test']], 
                        data_dir, 
                        pcp_dict=pcp_dict,
                        tmp_dict=tmp_dict,
                        conf_dict=conf_dict,
                        test_flag=test_flag)
            )
    return None


def build_series_from_dict(series_dict, row, series_length, num_series, num_years, normalizer, variable_name, unit='year'):
    '''
    Builds a series from a dictionary of monthly or yearly values
    series_dict:   dictionnary with stored values
    row:           observation from dataset
    series_length: number of values per series
    num_series:    number of series from variable
    num_years:     number of years taken in total
    normalizer:    dict with mean and std of variables' values over the dataset to normalize the series
    variable_name  key to get the mean and std in the normalizer dict
    unit:          'year' if yearly values, 'monthly' if monthly values in series
    '''
    # create empty series with proper dimensions
    monthly_series = np.zeros((series_length, num_series))
    yearly_mean_series = np.zeros((series_length, num_series))
    yearly_min_series = np.zeros((series_length, num_series))
    yearly_max_series = np.zeros((series_length, num_series))
    yearly_std_series = np.zeros((series_length, num_series))
    for year in range(row.year-num_years+1, row.year+1):
        # We recover yearly lists from dict
        start_index = year - (row.year-4)
        if unit=='month':
            # we fill the series with the 12 values for the current year
            monthly_series[0*(start_index+1):12*(start_index+1)] = np.array(series_dict[ (row.country, row.year, year, int(row.cluster)) ])
        else:
            
            yearly_series = np.array(series_dict[ (row.country, row.year, year, int(row.cluster)) ])
            yearly_mean_series[start_index] = np.mean(yearly_series)
            yearly_min_series[start_index] = np.min(yearly_series)
            yearly_max_series[start_index] = np.max(yearly_series)
            # yearly_std_series[start_index] = np.std(yearly_series)

    # we normalize using the dict normalizer[variable_name] mean and std
    if unit=='month':
        monthly_series = (monthly_series-normalizer[variable_name][0]) / normalizer[variable_name][1]
        return monthly_series
    else:
        #if variable_name=='precipitation':
         #   yearly_mean_series = (yearly_mean_series-normalizer[variable_name][0][0]) / normalizer[variable_name][1][0]
          #  yearly_min_series = (yearly_min_series-normalizer[variable_name][0][0]) / normalizer[variable_name][1][0]
           # yearly_max_series = (yearly_max_series-normalizer[variable_name][0][0]) / normalizer[variable_name][1][0]
            #return yearly_mean_series, yearly_min_series, yearly_max_series, yearly_std_series
            
        yearly_mean_series = (yearly_mean_series-normalizer[variable_name][0]) / normalizer[variable_name][1]
        yearly_min_series = (yearly_min_series-normalizer[variable_name][0]) / normalizer[variable_name][1]
        yearly_max_series = (yearly_max_series-normalizer[variable_name][0]) / normalizer[variable_name][1]
        return yearly_mean_series, yearly_min_series, yearly_max_series, yearly_std_series