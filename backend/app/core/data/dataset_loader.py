import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def _transform_for_dataset(dataset: str):
    if dataset == "mnist":
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose([transforms.ToTensor()])


def get_dataset(dataset: str, train: bool = False):
    transform = _transform_for_dataset(dataset)
    if dataset == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    return torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)


def get_loader(dataset: str, sample_limit: int, batch_size: int):
    ds = get_dataset(dataset, train=False)
    indices = list(range(min(sample_limit, len(ds))))
    subset = Subset(ds, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False), indices
