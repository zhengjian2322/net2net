import torchvision
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from torch.utils.data import DataLoader


def data_loader(train_batch_size, test_batch_size):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    pre_process = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        Cutout(),
        transforms.ToTensor(),

        normalize
    ]
    transform_train = transforms.Compose(pre_process)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_data = torchvision.datasets.CIFAR10(
        root='dataset/',
        train=True,
        transform=transform_train,
        download=False,
    )

    test_data = torchvision.datasets.CIFAR10(
        root='dataset/',
        train=False,
        transform=transform_test,
        download=False
    )
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
