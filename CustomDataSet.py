import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import get_augmentor, split_xml_path
from matplotlib import pyplot as plt


VOC_COLOR_MAP = {
    (0, 0, 0): 0,       # 배경
    (128, 0, 0): 1,     # Aeroplane
    (0, 128, 0): 2,     # Bicycle
    (128, 128, 0): 3,   # Bird
    (0, 0, 128): 4,     # Boat
    (128, 0, 128): 5,   # Bottle
    (0, 128, 128): 6,   # Bus
    (128, 128, 128): 7, # Car
    (64, 0, 0): 8,      # Cat
    (192, 0, 0): 9,     # Chair
    (64, 128, 0): 10,   # Cow
    (192, 128, 0): 11,  # Dining Table
    (64, 0, 128): 12,   # Dog
    (192, 0, 128): 13,  # Horse
    (64, 128, 128): 14, # Motorbike
    (192, 128, 128): 15, # Person
    (0, 64, 0): 16,     # Potted Plant
    (128, 64, 0): 17,   # Sheep
    (0, 192, 0): 18,    # Sofa
    (128, 192, 0): 19,  # Train
    (0, 64, 128): 20    # TV/Monitor
}


def convert_to_class_mask(mask_path):
    mask = Image.open(mask_path).convert('RGB')
    mask = np.array(mask)
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

    for color, class_id in VOC_COLOR_MAP.items():
        class_mask[(mask == color).all(axis=2)] = class_id

    return class_mask


class CustomDataSet(Dataset):
    def __init__(self, img_dir, label_dir):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.img = torch.from_numpy(np.zeros((len(self.img_dir), 3, 224, 224))).float()
        self.label = torch.from_numpy(np.zeros((len(self.label_dir), 224, 224))).float()

        for i, (img_one_path, label_one_path) in enumerate(zip(self.img_dir, self.label_dir)):
            img = Image.open(img_one_path).convert('RGB')
            label = convert_to_class_mask(label_one_path)  # 정수형 마스크
            img = np.array(img)

            # 이미지를 224x224로 리사이즈
            img = cv2.resize(img, dsize=(224, 224))

            # 라벨을 부동소수점으로 변환 후 리사이즈
            label = cv2.resize(label.astype(np.float32), dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            label = label.astype(np.int64)  # 다시 정수형으로 변환

            self.img[i] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            self.label[i] = torch.from_numpy(label).float()

        print(f"전체 : {len(self.img_dir)}")

    def __getitem__(self, idx):
        img, label = self.img[idx], self.label[idx]
        return img, label.unsqueeze(0)

    def __len__(self):
        return len(self.img_dir)


if __name__ == '__main__':
    (X_train_path, Y_train_path), (X_test_path, Y_test_path), (X_val_path, Y_val_path), (X_sample_path, Y_sample_path) = split_xml_path()
    sample_ds = CustomDataSet(X_sample_path, Y_sample_path)
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

