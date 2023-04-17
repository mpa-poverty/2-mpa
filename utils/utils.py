import torch


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
