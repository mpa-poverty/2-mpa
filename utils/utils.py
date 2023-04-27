import torch
import numpy as np

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