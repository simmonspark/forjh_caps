import torch
from torch.utils.data import DataLoader
from model import UNet
from CustomDataSet import CustomDataSet
from utils import split_xml_path
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            predicted_masks = (predictions > 0.5).float()
            visualize_results(images, masks, predicted_masks, idx)

def visualize_results(images, masks, predicted_masks, idx):
    images = images.cpu()
    masks = masks.cpu()
    predicted_masks = predicted_masks.cpu()

    for i in range(len(images)):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(images[i].permute(1, 2, 0).numpy())
        ax[0].set_title("Input Image")

        ax[1].imshow(masks[i].squeeze(), cmap="gray")
        ax[1].set_title("True Mask")

        ax[2].imshow(predicted_masks[i].squeeze(), cmap="gray")
        ax[2].set_title("Predicted Mask")

        plt.show()


# 테스트 코드 실행
if __name__ == "__main__":
    _, (X_test_path, Y_test_path), _, _ = split_xml_path()
    test_ds = CustomDataSet(X_test_path, Y_test_path)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    model_path = "trained_unet.pth"
    model = load_model(model_path)
    test_model(model, test_loader)
