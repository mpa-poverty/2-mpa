import torch


def test(model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        r2
        ):
    
    # Put model in eval mode
    model.eval() 
    device: torch.device
    
    test_r2 = 0, 0
    Y_true, Y_pred = None, None
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.float(), y.float()
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)
            
            # Registrate values for R2 estimation
            if Y_true is None:
                Y_true = y.view(-1,1)
                Y_pred = y_pred
            else:
                Y_true = torch.cat((Y_true, y.view(-1,1)))
                Y_pred = torch.cat((Y_pred, y_pred))

    # R2
    test_r2 = r2( Y_true.flatten(), Y_pred.flatten() )
    if device == 'cpu':
        return test_r2, Y_true.detach().numpy(), Y_pred.detach().numpy()
    return test_r2, Y_true.to('cpu').detach().numpy(), Y_pred.to('cpu').detach().numpy()
    
