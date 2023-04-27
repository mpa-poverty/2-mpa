import torch
import numpy as np
from typing import Dict, List
from scipy.stats import pearsonr





def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device:torch.device,
               r2):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss = 0
    # Loop through data loader data batches
    score = []
    for batch, (X, y) in enumerate(dataloader):        
        # Send data to target device
        X, y = X.float(), y.float()
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y.view(-1,1))
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        score.append(r2(y_pred, y.view(-1,1)))

    train_loss = train_loss / len(dataloader)
    total_score = sum(score)/len(score)
    return train_loss, total_score



def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              r2):
    # Put model in eval mode
    model.eval() 
    
    score=[]
    # Setup test loss and test accuracy values
    test_loss = 0
    # Turn on inference context manager
    # with torch.inference_mode():
  # Loop through DataLoader batches
    with torch.inference_mode():

      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.float(), y.float()
          X, y = X.to(device), y.to(device)
          # 1. Forward pass
          y_pred = model(X)
          # 2. Calculate and accumulate loss
          loss = loss_fn(y_pred, y.view(-1,1))
          test_loss += loss.item()
          
          score.append(r2(y_pred, y.view(-1,1)))

    total_score = sum(score)/len(score)
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss, total_score



def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          batch_size: int,
          in_channels: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          r2
        ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and val_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      val_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: SummaryWriter instance.
      
    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_r2": [],
               "test_loss": [],
               "test_r2": []
    }
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_r2 = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           r2=r2)
        test_loss, test_r2 = val_step(model=model,
                                      dataloader=val_dataloader,
                                      loss_fn=loss_fn,
                                      device=device,
                                      r2=r2)
        scheduler.step(test_loss)
        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_r2: {train_r2:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_r2: {test_r2:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_r2"].append(train_r2)
        results["test_loss"].append(test_loss)
        results["test_r2"].append(test_r2)

        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        # Add accuracy results to SummaryWriter
        writer.add_scalars(main_tag="R2", 
                           tag_scalar_dict={"train_r2": train_r2,
                                            "test_r2": test_r2}, 
                           global_step=epoch)
        
        # Track the PyTorch model architecture
        writer.add_graph(model=model, 
                         # Pass in an example input
                         input_to_model=torch.randn(batch_size, in_channels, 224, 224).to(device))
    
    # Close the writer
    writer.close()
    
    ### End new ###

    # Return the filled results at the end of the epochs
    return results


def dual_train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device:torch.device,
               r2):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss = 0
    # Loop through data loader data batches
    score = []
    for batch, (x1, x2, y) in enumerate(dataloader):        
        model.train()
        
        # Send data to target device
        x1, x2, y = x1.float(), x2.float(), y.float()
        x1, x2, y =  x1.to(device), x2.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(x1, x2)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y.view(-1,1))
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        score.append(r2(y_pred, y.view(-1,1)))

    train_loss = train_loss / len(dataloader)
    total_score = sum(score)/ len(score)
    return train_loss, total_score



def dual_val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              r2):
    # Put model in eval mode
    model.eval() 

    score=[]
    # Setup test loss and test accuracy values
    test_loss = 0
    # Turn on inference context manager
    # with torch.inference_mode():
  # Loop through DataLoader batches
    with torch.no_grad():

      for batch, (x1,x2, y) in enumerate(dataloader):
          
          # Send data to target device
          x1, x2, y = x1.float(), x2.float(), y.float()
          x1, x2, y =  x1.to(device), x2.to(device), y.to(device)
          # 1. Forward pass
          y_pred = model(x1,x2)
          # 2. Calculate and accumulate loss
          loss = loss_fn(y_pred, y.view(-1,1))
          test_loss += loss.item()
          
          score.append(r2(y_pred, y.view(-1,1)))
    total_score = sum(score)/len(score)
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss, total_score



def dual_train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          batch_size: int,
          in_channels: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          r2
        ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and val_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      val_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: SummaryWriter instance.
      
    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_r2": [],
               "test_loss": [],
               "test_r2": []
    }
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_r2 = dual_train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           r2=r2)
        test_loss, test_r2 = dual_val_step(model=model,
                                      dataloader=val_dataloader,
                                      loss_fn=loss_fn,
                                      device=device,
                                      r2=r2)
        scheduler.step(test_loss)
        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_r2: {train_r2:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_r2: {test_r2:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_r2"].append(train_r2)
        results["test_loss"].append(test_loss)
        results["test_r2"].append(test_r2)

        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        # Add accuracy results to SummaryWriter
        writer.add_scalars(main_tag="R2", 
                           tag_scalar_dict={"train_r2": train_r2,
                                            "test_r2": test_r2}, 
                           global_step=epoch)
        
        # Track the PyTorch model architecture
        # writer.add_graph(model=model, 
        #                  # Pass in an example input
        #                  input_to_model=torch.randn(batch_size, in_channels, 256, 256).to(device))
    
    # Close the writer
    writer.close()
    
    ### End new ###

    # Return the filled results at the end of the epochs
    return results