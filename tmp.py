import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import UNet, UNetSmall
from CustomDataSet import CustomDataSet
from utils import *
from knowledge_distilation import DistillationLoss
from scipy.stats import pearsonr
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_student(train_loader, val_loader, teacher_model, epochs=20, patience=2):
    teacher_model.eval()  # Freeze teacher model during student training
    student_model = UNetSmall().to(DEVICE)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-5)
    loss_fn = DistillationLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Student Training Epoch {epoch + 1}/{epochs}")
        student_model.train()
        loop = tqdm(train_loader, leave=True)

        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            with torch.no_grad():
                teacher_output = teacher_model(images)

            student_output = student_model(images)
            loss = loss_fn(student_output, teacher_output, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # Validation step
        print("Validating...")
        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                teacher_output = teacher_model(images)
                student_output = student_model(images)
                loss = loss_fn(student_output, teacher_output, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(student_model.state_dict(), "student_small_unet_best.pth")
            print(f"Validation loss improved to {val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return student_model


def train_step(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        predictions = model(images)
        loss = loss_fn(predictions, masks.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        mean_loss.append(loss.item())

    epoch_loss = sum(mean_loss) / len(mean_loss)
    print(f"One epoch loss: {epoch_loss}")
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
    print(f"Validation loss: {val_loss}")
    return val_loss


def train_model_with_early_stopping(train_loader, val_loader, model, loss_fn, epochs, patience=2, save_path="best_model.pth"):
    """
    Train the model with early stopping.

    Args:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - model: The model to train.
    - loss_fn: Loss function.
    - epochs: Number of epochs.
    - patience: Number of epochs to wait for improvement before stopping.
    - save_path: Path to save the best model.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train_losses, val_losses = [], []

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training step
        train_loss = train_step(train_loader, model, optimizer, loss_fn)
        val_loss = validate_step(val_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved to {val_loss:.4f}. Model saved to {save_path}.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses


def train_teacher(train_loader, val_loader, epochs=20, patience=2):
    teacher_model = UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    train_model_with_early_stopping(
        train_loader, val_loader, teacher_model, loss_fn, epochs, patience, save_path="teacher_unet_best.pth"
    )
    print("Best teacher model saved as teacher_unet_best.pth")
    return teacher_model


def compare_inference_speed(test_loader, teacher_model, student_model):
    teacher_model.eval()
    student_model.eval()

    def measure_time(model):
        loop = tqdm(test_loader, leave=False)
        total_time = 0
        with torch.no_grad():
            for images, _ in loop:
                images = images.to(DEVICE)
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                model(images)
                end_time.record()

                torch.cuda.synchronize()
                total_time += start_time.elapsed_time(end_time)

        return total_time / len(test_loader)

    teacher_time = measure_time(teacher_model)
    student_time = measure_time(student_model)

    print(f"Teacher Inference Speed: {teacher_time:.2f} ms/image")
    print(f"Student Inference Speed: {student_time:.2f} ms/image")

def fine_tune_student(train_loader, val_loader, student_model, epochs=5, patience=2):
    print("Starting Fine-tuning of Student Model...")
    optimizer = optim.Adam(student_model.parameters(), lr=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    train_model_with_early_stopping(
        train_loader, val_loader, student_model, loss_fn, epochs, patience, save_path="student_finetuned_unet.pth"
    )
    print("Fine-tuning complete. Best model saved as student_finetuned_unet.pth.")
    return student_model

def calculate_correlation_and_loss(test_loader, model, loss_fn):
    model.eval()
    losses = []
    correlations = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images)
            loss = loss_fn(predictions, masks.float()).item()
            losses.append(loss)

            # Convert tensors to numpy arrays for correlation calculation
            pred_np = torch.sigmoid(predictions).cpu().numpy().flatten()
            mask_np = masks.cpu().numpy().flatten()

            correlation, _ = pearsonr(pred_np, mask_np)
            correlations.append(correlation)

    avg_loss = np.mean(losses)
    avg_correlation = np.mean(correlations)
    return avg_loss, avg_correlation

def pipeline():
    # Prepare dataset and loaders
    (X_train_path, Y_train_path), (X_test_path, Y_test_path), (X_val_path, Y_val_path), _ = split_xml_path()

    train_ds = CustomDataSet(X_train_path, Y_train_path)
    val_ds = CustomDataSet(X_val_path, Y_val_path)
    test_ds = CustomDataSet(X_test_path, Y_test_path)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Train teacher model
    teacher_model = train_teacher(train_loader, val_loader)

    # Train student model using knowledge distillation
    student_model = train_student(train_loader, val_loader, teacher_model)

    # Fine-tune student model
    student_model = fine_tune_student(train_loader, val_loader, student_model)

    # Evaluate teacher and student models
    loss_fn = nn.BCEWithLogitsLoss()

    print("Evaluating Teacher Model...")
    teacher_loss, teacher_corr = calculate_correlation_and_loss(test_loader, teacher_model, loss_fn)
    print(f"Teacher Model - Loss: {teacher_loss:.4f}, Correlation: {teacher_corr:.4f}")

    print("Evaluating Student Model...")
    student_loss, student_corr = calculate_correlation_and_loss(test_loader, student_model, loss_fn)
    print(f"Student Model - Loss: {student_loss:.4f}, Correlation: {student_corr:.4f}")

    # Compare inference speed
    compare_inference_speed(test_loader, teacher_model, student_model)

if __name__ == "__main__":
    pipeline()
