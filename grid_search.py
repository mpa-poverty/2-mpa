# GRID_SEARCH.PY
# 
# USAGE: python3 ./grid_search.py configs/MODEL_CONFIG_GS.JSON MODEL_TYPE
# DESCRIPTION: Trains 5 models for each configuration specified in the json file
#              (see 'configs' for config files architecture)
# @MDC, 2023


# IMPORTS
import json
import pickle
import sys
import pandas as pd
import numpy as np
import torch
import itertools
import torchmetrics
from models import build_models
from utils import utils
from training import train_ms, train_msnl, train_msnlt, train_ts
import torchinfo

# CONSTANTS : DATASET
DATA_DIR = 'data/landsat_7_less/'
DATASET = 'data/dataset_2013+.csv'
FOLDS = 'data/dhs_incountry_folds_2013+.pkl'
SCHEDULER = 'ReduceLROnPlateau'  # options: 'ReduceLROnPlateau' or 'CyclicLR'


def cross_val_training(
        model_type: str,
        batch_size: int,
        epochs: int,
        lr: float,
        decay: float,
        save_path: str,
        loss_fn,
        model_config,
        r2,
        device,
        results,
):
    """Trains 5 models given a single configuration

    Args:
        model_type (str): possible values in ["ms","nl","msnl"]
        FROM CONFIG FILE:
            batch_size (int): training batch size
            epochs (int): total epochs during training
            lr (float): learning rate during training
            decay (float): weight decay during training
            loss_fn (torch.nn): loss function for training, MSE by default.
        save_path (str): root path to save models state_dicts to
        data_config (_type_): default input data transforms
        model_config (_type_): model config dictionary 
        r2 (torch.nn): R2 torchmetrics 
        device (str): "cuda" if GPU is available else "cpu"
        results (dict): dictionary to store train & validation metrics

    Returns:
        results (dict): filled result dictionary
    """

    with open(FOLDS, 'rb') as f:
        fold_dict = pickle.load(f)
    dataset = pd.read_csv(DATASET)

    for fold in fold_dict:
        if model_type == "msnl":
            ms_ckpt = model_config["ms_ckpt"] + str(fold) + ".pth"
            nl_ckpt = model_config["nl_ckpt"] + str(fold) + ".pth"
            model = build_models.build_model(model_type, model_config, device, ms_ckpt=ms_ckpt, nl_ckpt=nl_ckpt)

        elif model_type == 'msnlt':
            ms_ckpt = model_config["ms_ckpt"] + str(fold) + ".pth"
            nl_ckpt = model_config["nl_ckpt"] + str(fold) + ".pth"
            ts_ckpt = model_config["ts_ckpt"] + str(fold) + ".pth"
            model = build_models.build_model(model_type, model_config, device, ms_ckpt=ms_ckpt, nl_ckpt=nl_ckpt,
                                             ts_ckpt=ts_ckpt)
        else:
            model = build_models.build_model(model_type, model_config, device, ms_ckpt=None, nl_ckpt=None)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

        if SCHEDULER == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, max_lr=model_config['lr'] * 5,
                                                          base_lr=model_config['lr'], cycle_momentum=False)
        elif SCHEDULER == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
                                                                   patience=4, threshold=0.0001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=0, eps=1e-08)

        train_dataset, val_dataset = utils.datasets_from_model_type(
            model_type=model_type,
            data=dataset,
            data_dir=DATA_DIR,
            fold=fold,
            fold_dict=fold_dict
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        print(f"Training on fold " + str(fold))
        if model_type == 'msnl':
            results[fold] = train_msnl.msnl_finetune(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                ckpt_path=save_path + '_' + str(fold) + "_",
                device=device,
                r2=r2
            )
        elif model_type == 'ts':
            results[fold] = train_ts.train(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                ckpt_path=save_path + '_' + str(fold) + "_",
                r2=r2,
            )
        elif model_type == 'msnlt':
            results[fold] = train_msnlt.finetune(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                ckpt_path=save_path + '_' + str(fold) + "_",
                r2=r2,
            )
        else:
            results[fold] = train_ms.train(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                ckpt_path=save_path + '_' + str(fold) + "_",
                r2=r2,
            )
        print('Best validation epoch: ', np.argmax(results[fold]["test_r2"]) + 1)
    return results


def parse_gridsearch_arguments():
    """Parses grid_search.py script arguments from the command line.

    Returns:
        tuple (str, str, str): model_config_file path, data_config_file path, model_type
    """
    args = sys.argv
    try:
        assert len(args) == 3
    except AssertionError:
        print("Please enter the two config filenames, the network type and the pre_trained flag")
    return str(args[1]), str(args[2])


def main(
        model_config_filename: str,
        model_type: str
):
    """Trains 5 models for each configuration specified in the model_config_file
       Stores a RESULT dict for each configuration, with model in ['A','B','C','D','E'] as keys,
       which contains train/validation values across epochs for each of these.

    Args:
        model_config_filename (str): SE
        data_config_filename (str): SE, default to 'configs/default_config.pkl MODEL_TYPE'
        model_type (str): possible values in ["ms","nl","msnl"]
    Returns:
        None
    """
    # INIT VARIABLES
    results = dict()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r2 = torchmetrics.R2Score().to(device=device)

    # READING CONFIGS & DATA
    with open(model_config_filename) as f:
        model_config = json.load(f)

    lr = model_config['lr']
    batch_size = model_config['batch_size']
    decay = model_config['decay']
    n_epoch = model_config['n_epochs']
    configs = [[lr, batch_size, decay]]

    for i in range(len(configs)):
        lr, batch_size, decay = tuple(configs[i])
        print("CURRENT CONFIG: lr={}, batch_size={}, decay={}".format(lr, batch_size, decay))

        # BUILD MODEL
        results = cross_val_training(
            model_type=model_type,
            save_path=model_config['checkpoint_path'],
            batch_size=batch_size,
            epochs=n_epoch,
            lr=lr,
            decay=decay,
            loss_fn=torch.nn.MSELoss(),
            model_config=model_config,
            r2=r2,
            device=device,
            results=dict()
        )
        with open(model_config['result_path'] + "_" + str(i) + ".pkl", 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return None


if __name__ == "__main__":
    model_config_filename, model_type = parse_gridsearch_arguments()
    main(
        model_config_filename,
        model_type
    )
