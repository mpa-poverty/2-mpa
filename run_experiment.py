import json
import os 

import torch
import pickle
import torchinfo
import torchvision
import torchmetrics
import torchgeo.models
import numpy as np
import pandas as pd
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter
from data_handlers.csv_dataset import CustomDatasetFromDataFrame
from utils import utils
from train import train
from test import test
from models.from_config import build_from_config





# CONSTANTS
FOLD_PATH=os.path.join('data','dhs_incountry_folds.pkl')
CONFIG_FILE = 'configs/resnet18_ms_e2e_l7_1e2.json'
CSV_PATH=os.path.join('data','geometry_less_dataset.csv')
DATA_DIR=os.path.join('data','landsat_tif','')
TRAIN_TRANSFORM = torch.nn.Sequential(
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize(
            mean=[42.7178, 42.9092, 43.2119, 42.8700, 42.7862, 42.7192, 42.8525],
            std =[104.3150, 104.7388, 105.4271, 104.6307, 104.5374, 104.3182, 104.5891]
            ),
        torchvision.transforms.ColorJitter(),
    )
TEST_TRANSFORM  = torch.nn.Sequential(
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.Normalize(
            mean=[42.7178, 42.9092, 43.2119, 42.8700, 42.7862, 42.7192, 42.8525],
            std =[104.3150, 104.7388, 105.4271, 104.6307, 104.5374, 104.3182, 104.5891]
            ),
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WRITER = SummaryWriter()

def run_experiment( 
                    config_file=CONFIG_FILE, 
                    csv_path=CSV_PATH, 
                    data_dir=DATA_DIR,
                    writer=WRITER
                   ): 
    csv = pd.read_csv(csv_path)
    with open( config_file ) as f:
        config = json.load(f)
    with open(FOLD_PATH, 'rb') as f:
        folds = pickle.load(f)
    results = dict()

    # Define Transforms
    # Spatially Aware Cross-Validation
    for fold in folds:
        # Index split
        print(f"Training on fold {fold} has begun.")
        train_split = folds[fold]['train'][:64]
        val_split = folds[fold]['val'][:16]
        test_split = folds[fold]['test'][:16]
        # CSV split
        train_df = csv.iloc[train_split]
        val_df = csv.iloc[val_split]
        test_df = csv.iloc[test_split]
        # Datasets
        train_dataset = CustomDatasetFromDataFrame(train_df, data_dir, transform=TRAIN_TRANSFORM )
        val_dataset = CustomDatasetFromDataFrame(val_df, data_dir, transform=TEST_TRANSFORM )
        test_dataset  = CustomDatasetFromDataFrame(test_df, data_dir, transform=TEST_TRANSFORM )
        # DataLoaders
        train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
    
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        # MODEL
        base_model = torchgeo.models.resnet18(weights=torchgeo.models.ResNet18_Weights.SENTINEL2_ALL_MOCO)
        model = build_from_config( base_model=base_model, config_file=config_file )
        model = model.to(DEVICE)
        # CONFIGURE LOSS, OPTIM
        loss_fn = utils.configure_loss( config )
        optimizer = utils.configure_optimizer( config, model )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        r2 = torchmetrics.R2Score()
        r2 = r2.to(DEVICE)
        torchinfo.summary(model=model, 
        input_size=(config['batch_size'], 7, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)           
        # TRAINING
        results[fold] = {
            'train': train(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                epochs=config['n_epochs'],
                batch_size=config['batch_size'],
                in_channels=config['in_channels'],
                writer=writer,
                device=DEVICE,
                r2=r2
            ),
            'test': test(
                model=model,
                dataloader=test_loader,
                device=DEVICE
            )
        }
        # Clear GPU cache memory

    torch.save(model.state_dict(), config['checkpoint_path'])
    return results

if __name__ == "__main__":
    results = run_experiment()
    print(results)
    
