import torch
from data import data_loader


def configure_optimizer(config):
    if config.optimizer in ("Adam", "adam"):
        return torch.optim.Adam(model.parameters(), lr=model.config.lr)
    elif config.optimizer in ("SGD", 'sgd'):
        return torch.optim.SGD(model.parameters(), lr=model.config.lr)
    else:
        raise KeyError(config.optimizer)


def configure_loss(config):
    if config.loss in ("mse", "l2"):
         return torch.nn.MSELoss()
    elif config.loss in ("mae", "l1"):
         return torch.nn.L1Loss()
    else:
        raise KeyError(config.loss)


def configure_data_loader(config):
    if config.data_loader in ("Baseline", "baseline"):
        return data_loader.BaselineDataLoader(config)
    else:
        raise KeyError(config.data_loader)
    