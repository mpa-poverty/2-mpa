import json

import torch
import torchvision
import pickle
from models.double_branch import DoubleBranchCNN
from utils import utils, transfer_learning as tl


def build_from_config( base_model :torch.nn.Module, config)-> torch.nn.Module:
    """Generates a torch.nn.Module network adapted to a preformatted config file.
       The config file is expected to be written in json 
       and be organized as showcased in the __configs__ folder

    Args:
        base_model (torch.nn.Module): base model to be built upon
        config  : dict from json file

    Returns:
        torch.nn.Module: prepared network
    """
    
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


def build_model(model_config, device):
    # Test Dataset and Dataloader
    base_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    ms_branch = build_from_config( base_model=base_model, config=model_config )
    if model_config["branches"] == 2:
        nl_branch = tl.update_single_layer(torchvision.models.resnet18())
        model = DoubleBranchCNN(b1=ms_branch, b2=nl_branch, output_features=1)
        model = model.to(device=device)
    elif model_config["branches"] == 3:
        nl_branch = tl.update_single_layer(torchvision.models.resnet18())
        # third_branch = None
        # model = TripleBranchCNN(b1=ms_branch, b2=nl_branch, output_features=1)
    else:
        model = ms_branch.to(device=device)
    return model