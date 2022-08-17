# https://github.com/ildoonet/unsupervised-data-augmentation
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, Dataset
import numpy as np
from stratified_sampler import StratifiedSampler

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, transform_default, transform_aug, cutout=0):
        self.dataset = dataset
        self.transform_default = transform_default
        self.transform_aug = transform_aug
        self.transform_cutout = CutoutDefault(cutout)

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        img1 = self.transform_default(img)
        img2 = self.transform_default(self.transform_aug(img))
        img2 = self.transform_cutout(img2)

        return img1, img2

    def __len__(self):
        return len(self.dataset)

def get_multi_dataloaders():
    # label data train
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # unlabel data
    transform_valid = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # label data test
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    """
    param:
        augment policy
        interpolation mode
        pad value
    """
    auto_aug = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)

    sup_trainset = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transform_train)
    unsup_trainset = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=None)
    test_set = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transform_test)

    # split train dataset
    """
    param
        n_split - split number
    """

    sss = StratifiedShuffleSplit(n_splits=1, test_size=4000, random_state=0)
    sss = sss.split(list(range(len(sup_trainset))), sup_trainset.targets)

    train_idx, valid_idx = next(sss)
    # obtain all trainset label
    train_labels = [sup_trainset.targets[idx] for idx in train_idx]
    # split datasets based on train_idx
    trainset = Subset(sup_trainset, train_idx)  # for supervised
    trainset.train_labels = train_labels

    otherset = Subset(unsup_trainset, valid_idx)  # for unsupervised
    otherset = UnsupervisedDataset(otherset, transform_valid, auto_aug)

    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
    #     sampler=None, drop_last=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
        sampler=StratifiedSampler(trainset.train_labels), drop_last=True)

    unsuploader = torch.utils.data.DataLoader(
        otherset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True,
        sampler=None, drop_last=True)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
        drop_last=False
    )
    return trainloader, unsuploader, testloader


if __name__ == "__main__":
    get_multi_dataloaders()