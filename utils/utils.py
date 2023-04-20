import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda


MAX_VALUE = 2.643941
MIN_VALUE = -1.3713919

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


def normalize_asset(asset, min_asset=MIN_VALUE, max_asset=MAX_VALUE):
    return (asset- min_asset) / (max_asset - min_asset)

def denormalize_asset(asset, min_asset=MIN_VALUE, max_asset=MAX_VALUE):
    return asset * (max_asset - min_asset) + min_asset

def free_gpu_cache():
    gpu_usage()                
    cuda.select_device(0)
    torch.cuda.empty_cache()
    cuda.close()
    cuda.select_device(0)
    gpu_usage()
    return 