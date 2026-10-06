"""Offline assets and sample-keyed augmentation permit exact trial restart."""
from exp5.runtime import ROOT
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def image_transforms(cfg):
    interpolation = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize(cfg['mean'], cfg['std'])
    train = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test = transforms.Compose([transforms.Resize(int(224 / cfg['crop_pct']), interpolation=interpolation),
                               transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return train, test


class EpochData(Dataset):
    def __init__(self, base, order, seed, epoch):
        self.base, self.order, self.seed, self.epoch = base, order, seed, epoch

    def __len__(self):
        return len(self.order)

    def __getitem__(self, position):
        index = int(self.order[position])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 1000000 + self.epoch * 50000 + index)
            return self.base[index]


def loader(dataset, seed):
    return DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2,
                      pin_memory=True, multiprocessing_context='spawn',
                      generator=torch.Generator().manual_seed(seed))


def assets(cfg):
    train_transform, test_transform = image_transforms(cfg)
    train = datasets.CIFAR100(ROOT / 'data', train=True, download=False, transform=train_transform)
    test = datasets.CIFAR100(ROOT / 'data', train=False, download=False, transform=test_transform)
    assert len(train) == 50000 and len(test) == 10000
    return train, test
