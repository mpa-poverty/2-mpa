import torch
from tqdm import tqdm
from typing import Dict, List


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        r2):
    # We fine tune the model
    # Freezing all but the last layer's weights

    train_loss = 0
    score = []

    # Loop through data loader data batches
    for batch, (x1, x2, x3, y) in enumerate(tqdm(dataloader)):
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        x1, x2, x3, y = x1.float(), x2.float(), x3.float(), y.float()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(x1, x2, x3)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y.view(-1, 1))
        train_loss += loss.item()

        # 3. Optimizer zero grad (Avoids gradient piling up in memory)
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        score.append(r2(y_pred, y.view(-1, 1)))

    train_loss = train_loss / len(dataloader)
    total_score = sum(score) / len(score)

    return train_loss, total_score


def val_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        r2
):

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0
    score = []

    # Turn on inference context manager
    # with torch.inference_mode():
    with torch.no_grad():  # TODO: comparer avec inference_mode()

        # Loop through DataLoader batches
        for batch, (x1, x2, x3, y) in enumerate(tqdm(dataloader)):

            # Send data to target device
            x1, x2, x3, y = x1.float(), x2.float(), x3.float(), y.float()
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(x1, x2, x3)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y.view(-1, 1))
            test_loss += loss.item()

            score.append(r2(y_pred, y.view(-1, 1)))

    total_score = sum(score) / len(score)

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)

    return test_loss, total_score


def finetune(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        ckpt_path: str,
        device: torch.device,
        r2
) -> Dict[str, List]:
    """Fine-tunes a late fusion of two pre-trained PyTorch models.
    Calculates, prints and stores evaluation metrics throughout. 
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
        torch.save(model.state_dict(), ckpt_path + str(int(epoch + 1)) + ".pth")

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_r2: {train_r2:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_r2: {test_r2:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_r2"].append(train_r2.detach().cpu().numpy())
        results["test_loss"].append(test_loss)
        results["test_r2"].append(test_r2.detach().cpu().numpy())
    ### End new ###

    # Return the filled results at the end of the epochs
    return results
