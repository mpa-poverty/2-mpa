import torchvision
from models.double_branch_CNN import DoubleBranchCNN
from models.triple_branch import TripleBranch
from models.fcn_time_series import FCN
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

def build_vit(device, ms_ckpt=None):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = tl.update_last_layer(model=model, out_features=1, vit=True)
    if ms_ckpt is not None:
        model.load_state_dict(torch.load(ms_ckpt))
    return model.to(device)

def build_fcn(device, model_config, ckpt=None):
    num_channels = model_config['num_channels']
    output_size = model_config['output_size']
    model = FCN(num_channels=num_channels,output_size=output_size)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    return model.to(device)

def build_triple_branch( device, branch_1, branch_2, branch_3, msnlt_ckpt=None, with_vit=False):
    model = TripleBranch( branch_1=branch_1, branch_2=branch_2, branch_3=branch_3, output_features=1, with_vit=with_vit )
    if msnlt_ckpt is not None:
        model.load_state_dict(torch.load(msnlt_ckpt))
    return model.to(device)

def build_model( model_type, model_config, device, ms_ckpt, nl_ckpt, fcn_ckpt=None, msnl_ckpt=None, msnlt_ckpt=None):
    match model_type:
        case "ms":
            return build_ms(config=model_config, device=device, ms_ckpt=ms_ckpt)
        case "nl": 
            return build_nl( device=device, nl_ckpt=nl_ckpt)
        case "msnl":
            ms = build_ms(config=model_config, device=device, ms_ckpt=ms_ckpt)
            nl = build_nl(device=device, nl_ckpt=nl_ckpt)
            return build_msnl( msnl_ckpt=msnl_ckpt, ms=ms, nl=nl, device=device )
        case "vit":
            vit = build_vit(device=device, ms_ckpt=ms_ckpt)
            return vit
        case "fcn":
            fcn = build_fcn(device=device,model_config=model_config, ckpt=fcn_ckpt)
            return fcn
        case "msnlt":
            # ms = build_ms(config=model_config, device=device, ms_ckpt=ms_ckpt)
            vit = build_vit(device=device, ms_ckpt=ms_ckpt)
            nl = build_nl(device=device, nl_ckpt=nl_ckpt)
            fcn = build_fcn(device=device, model_config=model_config, ckpt=fcn_ckpt)
            return build_triple_branch( device=device, branch_1=vit, branch_2=nl, branch_3=fcn, msnlt_ckpt=msnlt_ckpt, with_vit=True)
    return None