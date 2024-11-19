'''import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import UNet
from CustomDataSet import CustomDataSet
from utils import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        predictions = model(images)
        if predictions.shape != masks.shape:
            print(f"Shape mismatch: predictions shape {predictions.shape}, masks shape {masks.shape}")
        loss = loss_fn(predictions, masks.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    epoch_loss = sum(mean_loss) / len(mean_loss)
    print(f'One epoch loss: {epoch_loss}')
    return epoch_loss

def validate_step(val_loader, model, loss_fn):
    model.eval()
    mean_loss = []

    with torch.no_grad():
        loop = tqdm(val_loader, leave=True)
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images)
            loss = loss_fn(predictions, masks)

            loop.set_postfix(loss=loss.item())
            mean_loss.append(loss.item())

    val_loss = sum(mean_loss) / len(mean_loss)
    print(f'Validation loss: {val_loss}')
    return val_loss


# 전체 학습 함수 정의
def train_model(train_loader, val_loader, model, optimizer, loss_fn, epochs):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_step(train_loader, model, optimizer, loss_fn)
        val_loss = validate_step(val_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses


def DoTrain():
    model = UNet().to(DEVICE)
    (X_train_path, Y_train_path), (X_test_path, Y_test_path), (X_val_path, Y_val_path), (
    X_sample_path, Y_sample_path) = split_xml_path()

    train_ds = CustomDataSet(X_train_path, Y_train_path)
    val_ds = CustomDataSet(X_val_path, Y_val_path)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 20
    train_losses, val_losses = train_model(train_loader, val_loader, model, optimizer, loss_fn, epochs)

    torch.save(model.state_dict(), "trained_unet.pth")
    print("Model saved as trained_unet.pth")
    print("Training Complete")
    print("Final Training Loss:", train_losses[-1])
    print("Final Validation Loss:", val_losses[-1])

if __name__ == "__main__":
    DoTrain()'''