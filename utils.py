import os
from functools import wraps
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

def print_total_data_count(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        total_count = sum(len(paths[0]) for paths in result)
        print(f"총 데이터 개수: {total_count}")
        return result

    return wrapper

@print_total_data_count
def split_xml_path():
    img_dir = '/media/unsi/media/data/voc/VOC2012_train_val/VOC2012_train_val/JPEGImages'
    label_dir = '/media/unsi/media/data/voc/VOC2012_train_val/VOC2012_train_val/SegmentationObject'
    img_full_path = []
    label_full_path = []

    for file_path, _, file_name in os.walk(label_dir):
        for name in file_name:
            label_path = os.path.join(label_dir, name)
            img_name = os.path.splitext(name)[0] + '.jpg'
            img_path = os.path.join(img_dir, img_name)
            img_full_path.append(img_path)
            label_full_path.append(label_path)

    X_train_path, X_test_path, Y_train_path, Y_test_path = train_test_split(
        img_full_path, label_full_path, train_size=0.95, test_size=0.05, shuffle=False
    )
    X_train_path, X_val_path, Y_train_path, Y_val_path = train_test_split(
        X_train_path, Y_train_path, train_size=0.9, test_size=0.1, shuffle=False
    )
    X_val_path, X_sample_path, Y_val_path, Y_sample_path = train_test_split(
        X_val_path, Y_val_path, train_size=0.9, test_size=0.1, shuffle=False
    )
    return (X_train_path, Y_train_path), (X_test_path, Y_test_path), (X_val_path, Y_val_path), (X_sample_path, Y_sample_path)

def get_augmentor():
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])


def test_open_data():
    (X_train_path, Y_train_path), _, _, _ = split_xml_path()
    img_path = X_train_path[0]
    label_path = Y_train_path[0]

    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path).convert('L')

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap="gray")
    plt.title("Label")
    plt.axis("off")

    plt.show()

if __name__ == '__main__':
    split_xml_path()
    test_open_data()
