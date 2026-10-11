"""Read only CIFAR's training pickle; never instantiate its test-reading loader."""
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from exp11 import ROOT, RESULTS
from exp11.config import SEED, SPLIT_SEED, file_hash, array_hash, save_json


def read_training():
    with (ROOT / 'data/cifar-100-python/train').open('rb') as f:
        raw = pickle.load(f, encoding='bytes')
    images = raw[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.asarray(raw[b'fine_labels'])


def prepare():
    from sklearn.model_selection import train_test_split
    from exp11.cv_model import checkpoint_path
    _, labels = read_training()
    train, valid = train_test_split(np.arange(50000), train_size=45000,
                                   random_state=SPLIT_SEED, stratify=labels)
    split = dict(train=np.sort(train).astype(np.int64),
                 search_validation=np.sort(valid).astype(np.int64))
    with np.load(ROOT / 'exp10/results/cv_split.npz') as reference:
        for key, value in split.items():
            np.testing.assert_array_equal(value, reference[key])
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / 'cv_split.npz', **split)
    manifest = dict(split_seed=SPLIT_SEED,
                    indices_sha256={k: array_hash(v) for k, v in split.items()},
                    examples={k: len(v) for k, v in split.items()},
                    cifar_train_sha256=file_hash(ROOT / 'data/cifar-100-python/train'),
                    pretrained_sha256=file_hash(checkpoint_path()), offline=True,
                    official_evaluation_loaded=False)
    save_json(RESULTS / 'assets_manifest.json', manifest)
    return manifest


class CIFARTraining(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index])


def cv_datasets(pretrained_cfg):
    normalize = transforms.Normalize(pretrained_cfg['mean'], pretrained_cfg['std'])
    bicubic = transforms.InterpolationMode.BICUBIC
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=bicubic),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    validation_transform = transforms.Compose([
        transforms.Resize(int(224 / pretrained_cfg['crop_pct']), interpolation=bicubic),
        transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    images, labels = read_training()
    with np.load(RESULTS / 'cv_split.npz') as split:
        indices = split['train'].copy()
        internal = split['search_validation'].copy()
    order = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(SEED)).numpy()]
    return (Subset(CIFARTraining(images, labels, training_transform), order.tolist()),
            Subset(CIFARTraining(images, labels, validation_transform), internal.tolist()), order)
