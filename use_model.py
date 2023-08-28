

import sys
import torch
import json
import pandas as pd
from models.build_models import build_model
import utils


def predict_wealth(df, model, dataloader, device):
    results = dict()
    # Put model in eval mode
    model.eval() 
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        if model_type=='msnl':
            for _, (idx, x1, x2, y) in enumerate(dataloader):
                # Send data to target device
                x1, x2, y = x1.float(), x2.float(), y.float()
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_pred = model(x1, x2)
                # Requires batch_size = 1 in the dataloader
                results[idx] = y_pred
        if model_type=='msnlt':
            for _, (idx, x1, x2,x3, y) in enumerate(dataloader):
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
        for idx in results:
            df.at[int(idx.cpu().numpy()[()][0]),'predicted_wealth']=results[idx].cpu().numpy()[()][0][0]
    return df






def main( 
        model_config_filename:str, 
        model_type:str,
        dataset_path:str,
        image_path:str,
        series_path:str,
        ckpt_path='models/checkpoints/'
        ):

    # INIT VARIABLES
    ckpt_path += model_type+'_'
    df = pd.read_csv(dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # READING CONFIGS & DATA
    with open( model_config_filename ) as f:
        model_config = json.ckpt(f)
    
    for fold in ['A','B','C','D','E']:
        if model_type=="ms" or model_type=="vit":
            model = build_model(model_config=model_config,
                    device=device, 
                    ms_ckpt=ckpt_path+fold+".pth", 
                    nl_ckpt=None, 
                    model_type=model_type)
        elif model_type=="nl":
            print(ckpt_path+fold+".pth")
            model = build_model(model_config=model_config,
                    device=device, 
                    nl_ckpt=ckpt_path+fold+".pth", 
                    ms_ckpt=None, 
                    model_type=model_type)
        elif model_type=="ts":
            model = build_model(model_config=model_config,
                    device=device, 
                    model_type=model_type,
                    fcn_ckpt=model_config['checkpoint_path']+str(fold)+".pth",
                    ms_ckpt=None,
                    nl_ckpt=None)
        elif model_type=="msnl":
            model = build_model(model_config=model_config,
                    device=device, 
                    msnl_ckpt=ckpt_path+fold+".pth", 
                    nl_ckpt=model_config["nl_ckpt"]+fold+".pth", 
                    ms_ckpt=model_config["ms_ckpt"]+fold+".pth", 
                    model_type=model_type)
        elif model_type=="msnlt":
            model = build_model(model_config=model_config,
                    device=device, 
                    msnlt_ckpt=ckpt_path+fold+".pth", 
                    nl_ckpt=model_config["nl_ckpt"]+fold+".pth", 
                    ms_ckpt=model_config["ms_ckpt"]+fold+".pth", 
                    fcn_ckpt=model_config['fcn_ckpt']+fold+".pth",
                    model_type=model_type)
        
        predict_set = utils.predict_set_from_model_type(
            model_type=model_type,
            data=df,
            data_dir=image_path,
            series_dir=series_path,
        )      
        predict_loader = torch.utils.data.DataLoader(
                predict_set,
                batch_size=1,
                shuffle=False
            )
            
        return predict_wealth(df, model, predict_loader, device)
    return None


    
def parse_gridsearch_arguments():
    """Parses grid_search.py script arguments from the command line.

    Returns:
        tuple (str, str, str): model_config_file path, data_config_file path, model_type
    """
    args = sys.argv
    try:
        assert len(args)==3
    except AssertionError:
        print("Please enter the two config filenames, the network type and the pre_trained flag")
    return str(args[1]),str(args[2])


if __name__ == "__main__":
    model_config_filename, model_type = parse_gridsearch_arguments()
    main(
        model_config_filename, 
        model_type
    )


