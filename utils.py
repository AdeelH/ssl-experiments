import numpy as np
from tqdm import tqdm
import albumentations as A

import torch
import torchvision as tv

CLASS_NAMES = np.array([
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
])
NUM_CLASSES = len(CLASS_NAMES)

base_tf = A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))


def collate_fn(tuples):
    x = np.stack([x for x, _ in tuples])
    y = np.stack([y for _, y in tuples])
    return x, y


def get_datasets(subset_size=4000):

    train_ds = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=True,
        download=True,
        transform=tv.transforms.Lambda(np.array))
    val_ds = tv.datasets.CIFAR10(
        root='./data/cifar/',
        train=False,
        download=True,
        transform=tv.transforms.Lambda(np.array))

    inds = [[] for _ in CLASS_NAMES]
    with tqdm(enumerate(train_ds.targets)) as bar:
        for i, y in bar:
            inds[y].append(i)
    inds = np.array(inds)

    def get_subset_inds(inds, n, complement=False):
        if not complement:
            out_inds = inds[:, :n].flatten()
        else:
            out_inds = inds[:, n:].flatten()
        return out_inds

    train_subset_ds = torch.utils.data.Subset(
        train_ds, get_subset_inds(inds, subset_size // NUM_CLASSES))

    return train_ds, train_subset_ds, val_ds


def validate(model, criterion, val_dl, bar_label_prefix=''):

    torch.cuda.reset_peak_memory_stats()

    val_loss = 0.
    val_corrects = 0

    model.eval()
    bar_label = f'{bar_label_prefix}Validating'
    with torch.no_grad():
        with tqdm(enumerate(val_dl), desc=bar_label, total=len(val_dl)) as bar:
            for i, (x, y) in bar:
                x, y = apply_transform(x, y, tf=base_tf)

                x = x.cuda()
                out = model(x).detach().cpu()
                val_loss += criterion(out, y)

                preds = out.argmax(dim=-1)
                val_corrects += (preds == y).float().sum()

    val_acc = val_corrects / len(val_dl.sampler)
    return val_loss, val_acc


def apply_transform(x, y, tf=None):
    if tf is not None:
        xs = [tf(image=_x)['image'] for _x in x.copy()]
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2)
    if y is not None:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        y = y.long()
    return x, y
