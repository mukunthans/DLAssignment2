
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

class DataPreparer:
    def __init__(self, h_params, image_size, train_dir, val_dir):
        self.h_params = h_params
        self.image_size = image_size
        self.train_dir = train_dir
        self.val_dir = val_dir

    def get_train_transform(self):
        size = (self.image_size, self.image_size)
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_test_transform(self):
        size = (self.image_size, self.image_size)
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def stratified_split(self, dataset, ratio):
        train_idx, val_idx = [], []
        class_bounds = [
            (0, 999), (1000, 1999), (2000, 2999), (3000, 3999), (4000, 4998),
            (4999, 5998), (5999, 6998), (6999, 7998), (7999, 8998), (8999, 9998)
        ]
        for start, end in class_bounds:
            indices = list(range(start, end + 1))
            split_at = int(len(indices) * ratio)
            train_idx.extend(indices[:split_at])
            val_idx.extend(indices[split_at:])
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def get_datasets(self):
        train_transform = self.get_train_transform()
        test_transform = self.get_test_transform()
        full_train = datasets.ImageFolder(self.train_dir, transform=train_transform)
        train_set, val_set = self.stratified_split(full_train, 0.8)
        test_set = datasets.ImageFolder(self.val_dir, transform=test_transform)
        return train_set, val_set, test_set

    def get_loaders(self):
        train_set, val_set, test_set = self.get_datasets()
        batch = self.h_params["batch_size"]
        return {
            "train_loader": DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=self.h_params["num_workers"]),
            "val_loader": DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=self.h_params["num_workers"]),
            "test_loader": DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=self.h_params["num_workers"]),
            "train_len": len(train_set),
            "val_len": len(val_set),
            "test_len": len(test_set)
        }
