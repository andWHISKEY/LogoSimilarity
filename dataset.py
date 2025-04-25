import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class ContrastiveTransformations:
    """하나의 이미지를 두 개의 augmentation view로 반환"""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]

class LogoDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            image = Image.new("RGB", (224, 224))
        
        # 두 개의 증강된 뷰를 텐서로 변환
        view1, view2 = self.transform(image) if self.transform else [transforms.ToTensor()(image)] * 2
        patent_number = os.path.splitext(os.path.basename(path))[0]
        
        # 두 개의 뷰를 하나의 텐서로 반환
        return torch.stack([view1, view2]), patent_number
