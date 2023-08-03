import torch
import torchvision
import numpy as np

def get_mean_std(dataset, ratio=0.01):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=0)
    train = next(iter(dataloader))[0]  # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std

if __name__ == '__main__':
    
# cifar10
    train_dataset = torchvision.datasets.CIFAR10('./data/cifar10',
                                                train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10('./data/cifar10',
                                                train=False, download=True,
                                                transform=torchvision.transforms.ToTensor())

    train_mean, train_std = get_mean_std(train_dataset)

    test_mean, test_std = get_mean_std(test_dataset)

    print(train_mean, train_std)
    print(test_mean, test_std)
