import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceDataset(Dataset):
    """
    Датасет для Face Recognition.
    Аналог работы с images и labels в NN.py,
    только данные читаются с диска.

    Структура папок (после препроцессинга):
    data/processed/
        0000045/          ← идентификатор (label)
            001.jpg
            002.jpg
        0000099/
            001.jpg
            ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Собираем все пути и метки
        self.samples = []
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = self.class_to_idx[class_dir.name]
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((str(img_path), label))

    def __getitem__(self, idx):
        """Возвращает (изображение, метка) по индексу."""
        img_path, label = self.samples[idx]

        # Чтение с диска
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Аугментации
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.samples)


def get_default_transforms(mode='train'):
    """Аугментации для тренировки и теста."""
    if mode == 'train':
        return A.Compose([
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(4, 16),
                hole_width_range=(4, 16),
                fill=0,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])