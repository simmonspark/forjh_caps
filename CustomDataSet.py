import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import get_augmentor, split_xml_path
from matplotlib import pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.img = torch.from_numpy(np.zeros((len(self.img_dir), 3, 224, 224))).float()
        self.label = torch.from_numpy(np.zeros((len(self.label_dir), 1, 224, 224))).float()

        for i, (img_one_path, label_one_path) in enumerate(zip(self.img_dir, self.label_dir)):
            img = Image.open(img_one_path).convert('RGB')
            label = Image.open(label_one_path).convert('L')
            img = np.array(img)
            label = np.array(label)
            img = cv2.resize(img, dsize=(224, 224))
            label = cv2.resize(label, dsize=(224, 224))
            self.img[i] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            self.label[i] = torch.from_numpy(label).unsqueeze(0).float() / 255.0

        print(f"전체 : {len(self.img_dir)}")

    def __getitem__(self, idx):
        img, label = self.img[idx], self.label[idx]
        return img, label

    def __len__(self):
        return len(self.img_dir)

if __name__ == '__main__':
    (X_train_path, Y_train_path), (X_test_path, Y_test_path), (X_val_path, Y_val_path), (X_sample_path, Y_sample_path) = split_xml_path()
    sample_ds = CustomDataset(X_train_path, Y_train_path)
    dataloader = DataLoader(sample_ds, batch_size=1, shuffle=False)

    x, y = next(iter(dataloader))
    x = x[0]
    y = y[0]

    plt.imshow(x.permute(1, 2, 0).cpu().numpy())
    plt.title("Image")
    plt.show()

    plt.imshow(y.squeeze(0).cpu().numpy(), cmap="gray")
    plt.title("Label")
    plt.show()
