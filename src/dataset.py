# src/dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_train_transforms():
    """
    Transforms used for training data
    """
    train_transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness = 0.2,
        contrast = 0.2,
        saturation = 0.2,
        hue = 0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])
    return train_transform

def get_eval_transforms():
    """
    Transforms used for validation and test data
    """
    eval_transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
    return eval_transform

def get_dataloaders(data_dir, batch_size, num_workers=2):
    """
    Creates DataLoaders for train, validation, and test sets
    """

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=get_train_transforms()
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=get_eval_transforms()
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=get_eval_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
