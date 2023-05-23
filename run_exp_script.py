import os
import json
import pickle
import sys
import pandas as pd
import torch 
import torchmetrics
import torchvision
from models.from_config import build_from_config
from models.double_branch import DoubleBranchCNN
from data_handlers.csv_dataset import CustomDatasetFromDataFrame
from utils import utils, transfer_learning as tl
from train import train, dual_train

DATA_DIR = 'data/'

def cross_val_training(model, model_config, data_config, r2, device, results):
    
    with open(data_config['fold'], 'rb') as f:
        fold_dict = pickle.load(f)
    dataset = pd.read_csv(data_config['csv'])
    for fold in fold_dict:
        train_dataset = CustomDatasetFromDataFrame(dataset.iloc[fold_dict[fold]['train']], DATA_DIR, transform=data_config['train_transform'],tile_max=data_config['max_'],
                                        tile_min=data_config['min_'] )
        val_dataset = CustomDatasetFromDataFrame(dataset.iloc[fold_dict[fold]['val']], DATA_DIR, transform=data_config['test_transform'],tile_max=data_config['max_'],
                                        tile_min=data_config['min'] )     
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=model_config['batch_size'], 
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=model_config['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        loss_fn = utils.configure_loss( model_config )
        optimizer = utils.configure_optimizer( model_config, model )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        print(f"Training on fold "+str(fold))
        results[fold] = train(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=model_config['n_epochs'],
            device=device,
            ckpt_path=model_config['checkpoint_path']+"_"+str(fold)+".pth",
            r2=r2
        )
        with open( model_config['result_path'], "wb" ) as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        
    return results


def full_training(model, model_config, data_config, r2, device, results):
    train_csv = pd.read_csv('data/madagascar_train_dataset.csv')
    test_csv = pd.read_csv('data/madagascar_test_dataset.csv')
    train_set = CustomDatasetFromDataFrame(train_csv, DATA_DIR, transform=data_config['train_transform'],tile_max=data_config['max_'],
                                        tile_min=data_config['min_'] )
    test_set = CustomDatasetFromDataFrame(test_csv, DATA_DIR, transform=data_config['test_transform'],tile_max=data_config['max_'],
                                        tile_min=data_config['min'] )     
    train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=model_config['batch_size'], 
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    val_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=model_config['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    loss_fn = utils.configure_loss( model_config )
    optimizer = utils.configure_optimizer( model_config, model )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    results = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=model_config['n_epochs'],
        device=device,
        ckpt_path=model_config['checkpoint_path']+"_full"+".pth",
        r2=r2
    )
    with open( model_config['result_path'], "wb" ) as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def parse_arguments():
    args = sys.argv
    try:
        assert len(args)==4
    except AssertionError:
        print("Please enter the two config filenames, the network type and the pre_trained flag")
    return str(args[1]),str(args[2]),str(args[3])


def main( network_config_filename:str, 
          data_config_filename:str,
          cross_val:str
        ):
    """Main train / val script that runs from two configuration files.
    Args:
        net_config_filename (str): Name of the model config file
            ------------- expected architecture -----------------
            {
                learning_rates : [float],
                optimizer : str,
                loss : str,
                weight_decay : [float],
                batch_sizes : [float],
                n_epochs : [float] 
            }           
        data_config_filename (str): Name of the dataset config file
        ------------- expected architecture -----------------
            {
                # FIXED
                train_transform, test_transform : torch.nn.Sequential, _
                tile_min, tile_max : [float], _
                means, stds : torch.tensor, _
            }        
    """
    
    # INIT VARIABLES
    results = dict()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r2 = torchmetrics.R2Score().to(device=device)
    # READING CONFIGS & DATA
    with open( network_config_filename ) as f:
        model_config = json.load(f)
    with open( data_config_filename,'rb') as f:
        data_config = pickle.load(f)
    # BUILD MODEL
    base_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
    ms_branch = build_from_config( base_model=base_model, config_file=model_config )
    if model_config["dual"] == True:
        nl_branch = tl.update_single_layer(torchvision.models.resnet18())
        model = DoubleBranchCNN(b1=ms_branch, b2=nl_branch, output_features=1)
    else:
        model = ms_branch.to(device=device)
    # TRAIN / VAL
    if cross_val=="True" or cross_val=="1":
        results = cross_val_training(model, model_config, data_config, r2, device, results)
    else: 
        results = full_training(model, model_config, data_config, r2, device, results)
    return results


if __name__ == "__main__":
    net_config_filename, data_config_filename, network_type = parse_arguments()
    main(
        net_config_filename, 
        data_config_filename, 
        network_type, 
        )


