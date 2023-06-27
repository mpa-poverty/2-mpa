import torchvision
from models.double_branch_CNN import DoubleBranchCNN
from utils import transfer_learning as tl
import torchgeo.models
import torch

def build_ms( config, device ):
    """Returns an instance of MS ResNet18"""
    base_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    model = tl.update_first_layer(
            base_model, 
            in_channels=config['in_channels'], 
            weights_init=config['weights_init'],
            scaling=config['scaling']
        )
    return model.to(device)

def build_nl( device ):
    """Returns an instance of NL ResNet18"""
    model = tl.update_single_layer(torchvision.models.resnet18())
    return model.to(device)

def build_msnl( ms, nl, device ):
    """Returns an instance of MS ResNet18"""
    model = DoubleBranchCNN(ms, nl, output_features=1)
    return model.to(device)

def build_vit():
    model = torchgeo.models.vit_small_patch16_224(weights=torchgeo.models.ViTSmall16_Weights.SENTINEL2_ALL_DINO)
    return model



def build_model( model_type, model_config, device, ms_ckpt, nl_ckpt ):
    match model_type:
        case "ms":
            return build_ms(config=model_config, device=device)
        case "nl": 
            return build_nl( device=device )
        case "msnl":
            ms = build_ms(config=model_config, device=device).load_state_dict(torch.load(ms_ckpt))
            nl = build_nl(device=device).load_state_dict(torch.load(nl_ckpt))
            return build_msnl( ms, nl, device=device )
        case "vit":
            ms = build_vit()
            return ms
    return None