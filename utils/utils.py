import torch
from datetime import datetime, timedelta
import numpy as np
import sys

def configure_optimizer(model):
    if model.config['optimizer'] in ("Adam", "adam"):
        return torch.optim.Adam(model.model.parameters(), lr=model.config['learning_rate'], weight_decay=model.config['weight_decay'])
    elif model.config['optimizer'] in ("SGD", 'sgd'):
        return torch.optim.SGD(model.model.parameters(), lr=model.config['learning_rate'], weight_decay=model.config['weight_decay'])
    else:
        raise KeyError(model.config['optimizer'])


def configure_loss(model):
    if model.config['loss'] in ("mse", "l2"):
         return torch.nn.MSELoss()
    elif model.config['loss'] in ("mae", "l1"):
         return torch.nn.L1Loss()
    else:
        raise KeyError(model.config['loss'])

    


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

