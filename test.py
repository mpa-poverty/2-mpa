import torch
import numpy as np
from scipy.stats import pearsonr


def test(model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        device
        ):
    
    # Put model in eval mode
    model.eval() 
    score = []
    Y_true, Y_pred = np.array([]), np.array([])
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
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

            score.append(pearsonr(target, output)[0]**2)
            Y_true = np.concatenate((Y_true,target))
            Y_pred = np.concatenate((Y_pred,output))
    total_score = sum(score)/len(score)
    return total_score, Y_true, Y_pred
    
