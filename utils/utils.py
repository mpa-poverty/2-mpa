import torch
from data import data_loader
from datetime import datetime, timedelta


def configure_optimizer(config):
    if config.optimizer in ("Adam", "adam"):
        return torch.optim.Adam(model.parameters(), lr=model.config.lr, weight_decay=config.weight_decay)
    elif config.optimizer in ("SGD", 'sgd'):
        return torch.optim.SGD(model.parameters(), lr=model.config.lr, weight_decay=config.weight_decay)
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
    


def disambiguate_timestamps(year: float, month: float, day: float):
    """Disambiguate partial timestamps.
    Based on :func:`torchgeo.datasets.utils.disambiguate_timestamps`.

    Args:
        year: year, possibly nan
        month: month, possibly nan
        day: day, possibly nan

    Returns:    
        minimum and maximum possible time range
    """
    if np.isnan(year):
        # No temporal info
        return 0, sys.maxsize
    elif np.isnan(month):
        # Year resolution
        mint = datetime(int(year), 1, 1)
        maxt = datetime(int(year) + 1, 1, 1)
    elif np.isnan(day):
        # Month resolution
        mint = datetime(int(year), int(month), 1)
        if month == 12:
            maxt = datetime(int(year) + 1, 1, 1)
        else:
            maxt = datetime(int(year), int(month) + 1, 1)
    else:
        # Day resolution
        mint = datetime(int(year), int(month), int(day))
        maxt = mint + timedelta(days=1)

    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()

