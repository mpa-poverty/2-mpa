import torchvision
from models.double_branch_CNN import DoubleBranchCNN
from models.lstm_regressor import LSTMRegressor
from utils import transfer_learning as tl
import torchgeo.models
import torch
import timm

def build_ms( config, device, ms_ckpt=None ):
    """Returns an instance of MS ResNet18"""
    base_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    model = tl.update_last_layer(
                tl.update_first_layer(
                    base_model, 
                    in_channels=config['in_channels'], 
                    weights_init=config['weights_init'],
                    scaling=config['scaling']
                )
        )
    if ms_ckpt is not None:
        model.load_state_dict(torch.load(ms_ckpt))
    return model.to(device)

def build_nl( device, nl_ckpt ):
    """Returns an instance of NL ResNet18"""
    model = tl.update_last_layer(tl.update_single_layer(torchvision.models.resnet18()))
    if nl_ckpt is not None:
        model.load_state_dict(torch.load(nl_ckpt))
    return model.to(device)

def build_msnl( ms, nl, device, msnl_ckpt=None ):
    """Returns an instance of MS ResNet18"""
    if msnl_ckpt is not None:
        model = DoubleBranchCNN(ms, nl, output_features=1)
        model.load_state_dict(torch.load(msnl_ckpt))
        return model.to(device)
    model = DoubleBranchCNN(ms, nl, output_features=1)
    return model.to(device)

def build_vit(device):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = tl.update_last_layer(model=model, out_features=1, vit=True)
    return model.to(device)

def build_lstm(msnl, device):
    # input_size = 6  # Concatenated input size: 2 sequences x 3 features per time step
    input_size = 3
    hidden_size = 3
    num_layers = 5
    model = LSTMRegressor(msnl, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    return model.to(device)


def build_model( model_type, model_config, device, ms_ckpt, nl_ckpt, msnl_ckpt=None ):
    match model_type:
        case "ms":
            return build_ms(config=model_config, device=device, ms_ckpt=ms_ckpt)
        case "nl": 
            return build_nl( device=device, nl_ckpt=nl_ckpt)
        case "msnl":
            ms = build_ms(config=model_config, device=device, ms_ckpt=ms_ckpt)#.load_state_dict(torch.load(ms_ckpt))
            nl = build_nl(device=device, nl_ckpt=nl_ckpt)#.load_state_dict(torch.load(nl_ckpt))
            return build_msnl( msnl_ckpt=msnl_ckpt, ms=ms, nl=nl, device=device )
        case "vit":
            vit = build_vit(device=device)
            return vit
        case "lstm":
            lstm = build_lstm(device=device)
            return lstm
    return None