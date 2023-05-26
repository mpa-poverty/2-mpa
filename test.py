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
    model = build_model(model_config=model_config,
                        device=device)
    with open( network_config_filename ) as f:
            model_config = json.load(f)
    model.load_state_dict(torch.load(model_config['checkpoint_path']+"_full"+".pth"))
    # Dataset & Loader
    
    with open( data_config['fold'], 'rb') as f:
         fold_dict = pickle.load(f)
    dataset = dataset.iloc[fold_dict['A']['test']]
    dataset = dataset.reset_index()
    test_set = CustomTestDatasetFromDataFrame(dataset, DATA_DIR, transform=data_config['test_transform'],tile_max=data_config['max'],
                                        tile_min=data_config['min'] )
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
        )
    # Test result per row.index
    results = test(model, test_loader, device)
    for idx in results:
        dataset.at[idx,'predicted_wealth']=results[idx].cpu().numpy()[()]
    with open( write_path, "wb" ) as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset
# if __name__ == "__main__":
#      main()