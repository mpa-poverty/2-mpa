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
        r2
        ):
    results = dict()
    # Put model in eval mode
    model.eval() 
    Y_true, Y_pred = np.array([]), np.array([])
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (idx, X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.float(), y.float()
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)
            
            output=y_pred.view(-1,1).detach().cpu().numpy()
            target=y.detach().cpu().numpy()
            output=np.squeeze(output)
            output=np.nan_to_num(output)
            target=np.squeeze(target)
            target=np.nan_to_num(target)

            Y_true = np.concatenate((Y_true,target))
            Y_pred = np.concatenate((Y_pred,output))
            # Requires batch_size = 1 in the dataloader
            results[idx] = r2(Y_true, Y_pred)
    return results
    

def main(
          write_path:str,
          network_config_filename:str, 
          data_config_filename:str,
          dataset:pd.DataFrame
          )->dict:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r2 = torchmetrics.R2Score().to(device=device)
    # Load Model
    model = build_model(data_config_filename=data_config_filename,
                        network_config_filename=network_config_filename,
                        device=device)
    with open( network_config_filename ) as f:
            model_config = json.load(f)
    model.load_state_dict(torch.load(model_config['checkpoint_path']+"_full"+".pth"))
    # Dataset & Loader
    with open( data_config_filename,'rb') as f:
        data_config = pickle.load(f)
    test_set = CustomTestDatasetFromDataFrame(dataset.iloc[data_config['fold']['A']['test']], DATA_DIR, transform=data_config['test_transform'],tile_max=data_config['max'],
                                        tile_min=data_config['min'] )
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
        )
    # Test result per row.index
    results = test(model, test_loader, device, r2)
    dataset['predicted_wealth'] = np.NaN
    for idx in results:
        dataset.at[idx,'predicted_wealth']=results['idx'].cpu().numpy()[()]
    with open( write_path, "wb" ) as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset

# if __name__ == "__main__":
#      main()