import os
import json
import pickle
import sys
import pandas as pd
import torch 
import numpy as np
import torchmetrics
import torchvision
from models.from_config import build_model
from models.double_branch import DoubleBranchCNN
from data_handlers.csv_dataset import CustomDatasetFromDataFrame
from utils import utils, transfer_learning as tl
from train import train, dual_train
from sklearn.model_selection  import train_test_split

DATA_DIR = 'data/landsat_7/'

def cross_val_training(
        model, 
        batch_size,
        epochs,
        lr,
        decay,
        loss_fn,
        data_config,
        r2,
        device,
        results,
        save_path
        ):
    
    with open(data_config['fold'], 'rb') as f:
        fold_dict = pickle.load(f)
    dataset = pd.read_csv(data_config['csv'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    for fold in fold_dict:
        train_dataset = CustomDatasetFromDataFrame(dataset.iloc[fold_dict[fold]['train']], DATA_DIR, transform=data_config['train_transform'],tile_max=data_config['max'],
                                        tile_min=data_config['min'] )
        val_dataset = CustomDatasetFromDataFrame(dataset.iloc[fold_dict[fold]['val']], DATA_DIR, transform=data_config['test_transform'],tile_max=data_config['max'],
                                        tile_min=data_config['min'] )     
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        print(f"Training on fold "+str(fold))
        results[fold] = train(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
            ckpt_path=save_path+'_'+str(int(1000*lr))+'_'+str(batch_size)+'_'+str(int(100*decay))+'_'+str(fold)+".pth",
            r2=r2
        )

    return results

    
def parse_arguments():
    args = sys.argv
    try:
        assert len(args)==3
    except AssertionError:
        print("Please enter the two config filenames, the network type and the pre_trained flag")
    return str(args[1]),str(args[2])


def main( 
        network_config_filename:str, 
        data_config_filename:str,
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


    lr_list = model_config['lr_list']
    batch_size_list = model_config['batch_size_list']
    decay_list = model_config['decay_list']
    n_epoch = 100
    for config in zip(lr_list, batch_size_list, decay_list):
        lr, batch_size, decay = config
        print("CURRENT CONFIG: lr={}, batch_size={}, decay={}".format(lr, batch_size, decay))
        # BUILD MODEL
        model = build_model(model_config, device)
        results['config'] = config
        results['cross_val_results'] = dict()
        results['cross_val_results'] = cross_val_training(
            model, 
            batch_size,
            n_epoch,
            lr,
            decay,
            torch.nn.MSELoss(),
            data_config,
            r2,
            device,
            results['cross_val_results'],
            save_path=model_config['checkpoint_path']
        )
    with open(model_config['result_path']+".pkl") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    return results


if __name__ == "__main__":
    net_config_filename, data_config_filename = parse_arguments()
    main(
        net_config_filename, 
        data_config_filename, 
    )


