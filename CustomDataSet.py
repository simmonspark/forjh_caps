import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import get_augmentor, split_xml_path
from matplotlib import pyplot as plt


def convert_to_binary_mask(mask_path):
    mask = Image.open(mask_path).convert('RGB')
    mask = np.array(mask)
    binary_mask = np.where(mask.sum(axis=2) == 0, 1, 0)

    return torch.tensor(binary_mask, dtype=torch.float32)

class CustomDataSet(Dataset):
    def __init__(self, img_dir, label_dir):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.img = torch.from_numpy(np.zeros((len(self.img_dir), 3, 224, 224))).float()
        self.label = torch.from_numpy(np.zeros((len(self.label_dir), 1, 224, 224))).float()

        for i, (img_one_path, label_one_path) in enumerate(zip(self.img_dir, self.label_dir)):
            img = Image.open(img_one_path).convert('RGB')
            label = convert_to_binary_mask(label_one_path)
            img = np.array(img)
            label = np.array(label)
            img = cv2.resize(img, dsize=(224, 224))
            label = cv2.resize(label, dsize=(224, 224))
            self.img[i] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            self.label[i] = torch.from_numpy(label).unsqueeze(0).float()

        print(f"전체 : {len(self.img_dir)}")

    def __getitem__(self, idx):
        img, label = self.img[idx], self.label[idx]
        return img, label

    def __len__(self):
        return len(self.img_dir)

if __name__ == '__main__':
    (X_train_path, Y_train_path), (X_test_path, Y_test_path), (X_val_path, Y_val_path), (X_sample_path, Y_sample_path) = split_xml_path()
    sample_ds = CustomDataSet(X_train_path, Y_train_path)
    dataloader = DataLoader(sample_ds, batch_size=4, shuffle=False)

    x, y = next(iter(dataloader))
    print(x.shape,y.shape)
    x = x[0]
    y = y[0]

    plt.imshow(x.permute(1, 2, 0).cpu().numpy())
    plt.title("Image")
    plt.show()

    plt.imshow(y.squeeze(0).cpu().numpy(), cmap="gray")
    plt.title("Label")
    plt.show()

