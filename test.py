import torch
import numpy as np
import pandas as pd
import json
import pickle
import torchmetrics
from models.from_config import build_model
from data_handlers.csv_dataset import CustomTestDatasetFromDataFrame
from scipy.stats import pearsonr

DATA_DIR = 'data/landsat_7'


def test(model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        device,
        ):
    results = dict()
    # Put model in eval mode
    model.eval() 
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
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
        ):
    r2 = torchmetrics.R2Score().to(device=device)

    # Put model in eval mode
    model.eval() 
    score=[]
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.float(), y.float()
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            # Requires batch_size = 1 in the dataloader
            score.append(r2(y_pred, y.view(-1,1)))
    total_score = sum(score)/ len(score)
    return total_score
    

def main(
          write_path:str,
          network_config_filename:str, 
          data_config_filename:str,
          dataset:pd.DataFrame
          )->dict:
    with open( data_config_filename,'rb') as f:
        data_config = pickle.load(f)
    with open( network_config_filename,'rb') as f:
        model_config = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Model
    with open( network_config_filename ) as f:
            model_config = json.load(f)
    load_path = model_config['checkpoint_path']

    with open( data_config['fold'], 'rb') as f:
        fold_dict = pickle.load(f)


    for fold in ['A','B','C','D','E']:
        model = build_model(model_config=model_config,
                    device=device)
        model.load_state_dict(torch.load(load_path+fold+".pth"))
        dataset_ = dataset.iloc[fold_dict[fold]['test']]
        dataset_ = dataset_.reset_index()
        test_set = CustomTestDatasetFromDataFrame(dataset_, DATA_DIR, transform=data_config['test_transform'] )
        test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=1,
            )
        # Test result per row.index
        results = test(model, test_loader, device)
        for idx in results:
            dataset_.at[idx,'predicted_wealth']=results[idx].cpu().numpy()[()]
        dataset_.to_csv(write_path, index=False)
    return dataset_


if __name__ == "__main__":
     main()