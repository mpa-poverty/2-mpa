import torchvision
from models.double_branch_CNN import DoubleBranchCNN
from models.triple_branch import TripleBranch
from models.fcn_time_series import FCN
from utils import transfer_learning as tl
import torchgeo.models
import torch
import timm


def build_ms(device, ms_ckpt=None):
    """Returns an instance of MS ResNet18"""
    base_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    model = tl.update_last_layer(
        tl.update_first_layer(
            base_model,
            in_channels=7,
            weights_init='average',
            scaling=0.42
        )
    )
    if ms_ckpt is not None:
        model.load_state_dict(torch.load(ms_ckpt))
    return model.to(device)


def build_nl(device, nl_ckpt):
    """Returns an instance of NL ResNet18"""
    model = tl.update_last_layer(tl.update_single_layer(torchvision.models.resnet18()))
    if nl_ckpt is not None:
        model.load_state_dict(torch.load(nl_ckpt))
    return model.to(device)


def build_msnl(device, ms, nl, msnl_ckpt=None):
    if msnl_ckpt is not None:
        model = DoubleBranchCNN(ms, nl, output_features=1)
        model.load_state_dict(torch.load(msnl_ckpt))
        return model.to(device)
    model = DoubleBranchCNN(ms, nl, output_features=1)
    return model.to(device)


def build_ts(device, model_config, ts_ckpt=None):
    num_channels = model_config['num_channels']
    output_size = model_config['output_size']
    filter_size = model_config['filter_size']
    model = FCN(num_channels=num_channels, filter_size=filter_size, output_size=output_size)
    if ts_ckpt is not None:
        model.load_state_dict(torch.load(ts_ckpt))
    return model.to(device)


def build_msnlt(device, ms, nl, ts, msnlt_ckpt=None):
    model = TripleBranch(ms=ms, nl=nl, ts=ts, output_features=1)
    if msnlt_ckpt is not None:
        model.load_state_dict(torch.load(msnlt_ckpt))
    return model.to(device)


def build_model(model_type, model_config, device, ms_ckpt=None, nl_ckpt=None, ts_ckpt=None, msnl_ckpt=None,
                msnlt_ckpt=None):
    match model_type:
        case "ms":
            return build_ms(device=device, ms_ckpt=ms_ckpt)
        case "nl":
            return build_nl(device=device, nl_ckpt=nl_ckpt)
        case "msnl":
            ms = build_ms(device=device, ms_ckpt=ms_ckpt)
            nl = build_nl(device=device, nl_ckpt=nl_ckpt)
            return build_msnl(device=device, ms=ms, nl=nl, msnl_ckpt=msnl_ckpt)
        case "ts":
            ts = build_ts(device=device, model_config=model_config, ts_ckpt=ts_ckpt)
            return ts
        case "msnlt":
            ms = build_ms(device=device, ms_ckpt=ms_ckpt)
            nl = build_nl(device=device, nl_ckpt=nl_ckpt)
            ts = build_ts(device=device, model_config=model_config, ts_ckpt=ts_ckpt)
            return build_msnlt(device=device, ms=ms, nl=nl, ts=ts, msnlt_ckpt=msnlt_ckpt)

    return None
