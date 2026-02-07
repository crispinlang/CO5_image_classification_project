import torch, os

def set_device():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch):", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device



from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def get_transform_data():

    transform = transforms.Compose([
    transforms.Resize((256, 224)),
    transforms.ToTensor(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

    dataset = datasets.ImageFolder(
        root="archive/merged_dataset",
        transform=transform
    )

    N = len(dataset)

    train_size = int(0.8 * N)
    val_size   = int(0.1 * N)
    test_size  = N - train_size - val_size  # Rest for test

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(len(dataset))
    print(len(train_data), len(val_data), len(test_data))

    train_data = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_data = DataLoader(val_data, batch_size=256, shuffle=False)
    test_data = DataLoader(test_data, batch_size=256, shuffle=False)


    return train_data, val_data, test_data



import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1   = nn.BatchNorm2d(16) # added batch norm

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2   = nn.BatchNorm2d(32) # added batch norm

        self.res1 = ResBlock(32)   # added resudial blocks
        self.res2 = ResBlock(32)   # added resudial blocks

        self.fc1 = nn.Linear(103456, 120) # small model: 51728
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 169)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.res1(x)  # added resudial blocks
        x = self.res2(x)  # added resudial blocks
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

import matplotlib.pyplot as plt

def train_net(n_epoch, criterion, optimizer, device, net, train_data):
    losses = []
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.10f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    plt.plot(losses, label='Training loss')
    plt.xlabel("training steps")
    plt.ylabel("loss value")
    plt.show()
    print('Finished Training')
    return net


def evaluate(net, test_loader, device):
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"Accuracy on {total} test images: {acc:.2f}%")
    return acc


