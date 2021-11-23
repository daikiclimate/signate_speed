from .drive_dataset import DriveDataset
import torch

# from .transform import return_img_transform

from torch.utils.data import DataLoader


def return_dataset(config):
    dataset_type = config.type
    transforms = None

    train_dataset = DriveDataset(mode="train", transform=transforms)
    test_dataset = DriveDataset(mode="val", transform=transforms)

    return train_dataset, test_dataset


def return_test_dataset(config):
    test_dataset = DriveDataset(mode="test")
    return test_dataset


def return_dataloader(config):
    train_set, test_set = return_dataset(config)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        # shuffle=False,
        shuffle=True,
        drop_last=True,
        num_workers=config.n_worker,
        # pin_memory = False,
        # num_workers=8,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, drop_last=True, num_workers=8, shuffle=False
    )
    return train_loader, test_loader
