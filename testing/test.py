import torch
import pandas as pd
import json
import pickle
import torchmetrics
import numpy as np
from models.build_models import build_model
from utils import utils

DATA_DIR = 'data/landsat_7_less'


def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         device,
         model_type
         ):
    results = dict()
    # Put model in eval mode
    model.eval()
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        if model_type == 'msnl':
            for _, (idx, x1, x2, y) in enumerate(dataloader):
                # Send data to target device
                x1, x2, y = x1.float(), x2.float(), y.float()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_pred = model(x1, x2)
                # Requires batch_size = 1 in the dataloader
                results[idx] = y_pred
        elif model_type == 'msnlt':
            for _, (idx, x1, x2, x3, y) in enumerate(dataloader):
                # Send data to target device
                x1, x2, x3, y = x1.float(), x2.float(), x3.float(), y.float()
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                y_pred = model(x1, x2, x3)
                # Requires batch_size = 1 in the dataloader
                results[idx] = y_pred
        else:
            for _, (idx, X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.float(), y.float()
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                # Requires batch_size = 1 in the dataloader
                results[idx] = y_pred

    return results


def test_r2(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device,
            model_type
            ):
    r2 = torchmetrics.R2Score().to(device=device)

    # Put model in eval mode
    model.eval()
    score = []
    # Turn on inference context manager
    with torch.inference_mode():
        if model_type == "msnl":
            # Loop through DataLoader batches
            for batch, (x1, x2, y) in enumerate(dataloader):
                # Send data to target device
                x1, x2, y = x1.float(), x2.float(), y.float()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_pred = model(x1, x2)
                # Requires batch_size = 1 in the dataloader
                score.append(r2(y_pred, y.view(-1, 1)))
        elif model_type == "msnlt":
            # Loop through DataLoader batches
            for batch, (x1, x2, x3, y) in enumerate(dataloader):
                # Send data to target device
                x1, x2, x3, y = x1.float(), x2.float(), x3.float(), y.float()
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                y_pred = model(x1, x2, x3)
                # Requires batch_size = 1 in the dataloader
                score.append(r2(y_pred, y.view(-1, 1)))
        else:
            # Loop through DataLoader batches
            for _, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.float(), y.float()
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                # Requires batch_size = 1 in the dataloader
                score.append(r2(y_pred, y.view(-1, 1)))
        total_score = sum(score) / len(score)
    return total_score


def main(
        write_path: str,
        network_config_filename: str,
        fold_path: str,
        dataset: pd.DataFrame,
        model_type: str,
) -> dict:
    with open(network_config_filename, 'rb') as f:
        model_config = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    load_path = model_config['checkpoint_path']

    with open(fold_path, 'rb') as f:
        fold_dict = pickle.load(f)

    for fold in ['A', 'B', 'C', 'D', 'E']:
        # model = torch.load(load_path + fold + ".pth")
        # print(load_path + fold + ".pth")
        if model_type == "ts":
            model = build_model(model_config=model_config,
                                device=device,
                                ts_ckpt=load_path + fold + ".pth",
                                nl_ckpt=None,
                                ms_ckpt=None,
                                model_type=model_type)
        if model_type == "ms" or model_type == "vit":
            print(load_path + fold + ".pth")
            model = build_model(model_config=model_config,
                                device=device,
                                ms_ckpt=load_path + fold + ".pth",
                                nl_ckpt=None,
                                model_type=model_type)
        elif model_type == "nl":
            print(load_path + fold + ".pth")
            model = build_model(model_config=model_config,
                                device=device,
                                nl_ckpt=load_path + fold + ".pth",
                                ms_ckpt=None,
                                model_type=model_type)
        elif model_type == "fcn":
            model = build_model(model_config=model_config,
                                device=device,
                                model_type=model_type,
                                #fcn_ckpt=model_config['checkpoint_path']+str(fold)+".pth",
                                ms_ckpt=None,
                                nl_ckpt=None)
        elif model_type == "msnl":
            model = build_model(model_config=model_config,
                                device=device,
                                msnl_ckpt=load_path + fold + ".pth",
                                nl_ckpt=model_config["nl_ckpt"] + fold + ".pth",
                                ms_ckpt=model_config["ms_ckpt"] + fold + ".pth",
                                model_type=model_type)
        elif model_type == "msnlt":
            print(load_path + fold + ".pth")
            model = build_model(model_config=model_config,
                                device=device,
                                msnlt_ckpt=load_path + fold + ".pth",
                                nl_ckpt=model_config["nl_ckpt"] + fold + ".pth",
                                ms_ckpt=model_config["ms_ckpt"] + fold + ".pth",
                                #fcn_ckpt=model_config['fcn_ckpt']+fold+".pth",
                                model_type=model_type)

        test_set = utils.testset_from_model_type(
            model_type=model_type,
            data=dataset,
            data_dir=DATA_DIR,
            fold=fold,
            fold_dict=fold_dict
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False
        )
        # Test result per row.index
        results = test(model, test_loader, device, model_type=model_type)
        for idx in results:
            dataset.at[int(fold_dict[fold]['train'][idx.cpu().numpy()[()][0]]), 'predicted_wealth'] = \
                results[idx].cpu().numpy()[()][0][0]
    dataset.to_csv(write_path, index=False)
    return dataset


if __name__ == "__main__":
    main()
