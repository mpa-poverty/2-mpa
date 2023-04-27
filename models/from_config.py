import json

import torch

from utils import utils, transfer_learning as tl


def build_from_config( base_model :torch.nn.Module, config_file :str )-> torch.nn.Module:
    """Generates a torch.nn.Module network adapted to a preformatted config file.
       The config file is expected to be written in json 
       and be organized as showcased in the __configs__ folder

    Args:
        base_model (torch.nn.Module): base model to be built upon
        config_file (str): name of the config file

    Returns:
        torch.nn.Module: prepared network
    """
    
    with open(config_file) as f:
        config = json.load(f)
    
    if not config['transfer']:
        model = tl.update_first_layer(
            base_model, 
            in_channels=config['in_channels'], 
            weights_init=config['weights_init'],
            scaling=config['scaling']
        )
    else:
        
        model = tl.s2_to_landsat(base_model, scaling=config['scaling'])
        for param in model.parameters():
            param.requires_grad = False

    if not config["dual"]:
        model = tl.update_last_layer(model, config['out_features'])

    return model
