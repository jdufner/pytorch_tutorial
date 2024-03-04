# MNIST
# Dataloader, Transformation
# Multilayer Neural Net, Activation Function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import os
import torch
from torch import cuda
from torch import device
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision.datasets import MNIST
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from typing import Iterator, Tuple
import matplotlib as plt

# device config
device: device = device('cuda' if cuda.is_available() else 'cpu')

# hyper parameters
input_size: int = 784  # 28 x 28
hidden_size: int = 100
num_classes: int = 10
num_epochs: int = 2
batch_size: int = 100
learning_rate: float = .001

# MNIST
# Training + validation data
entire_dataset: VisionDataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_size: int = int(.8 * len(entire_dataset))
val_size: int = len(entire_dataset) - train_size
train_dataset, val_dataset = random_split(entire_dataset, [train_size, val_size])
train_data_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=os.cpu_count() // 2, persistent_workers=True)
val_data_loader: DataLoader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=os.cpu_count() // 2, persistent_workers=True)

# Test data
test_dataset: torchvision.datasets = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_data_loader: DataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=os.cpu_count() // 2, persistent_workers=True)

# without type hints
examples = iter(train_data_loader)
samples, labels = next(examples)
# print(samples.shape, labels.shape)

# with type hints
# examples: Iterator[Tuple[Tensor, Tensor]] = iter(train_data_loader)
# example: Tuple[Tensor, Tensor] = next(examples)
# samples: Tensor = example[0]
# labels: Tensor = example[1]

# for i in range(6):  # type: int
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()


class NeuralNet(Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1: Linear = Linear(input_size, hidden_size)
        self.relu: ReLU = ReLU()
        self.l2: Linear = Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> Linear:
        out: Linear = self.l1(x)
        out: ReLU = self.relu(out)
        out: Linear = self.l2(out)
        return out


model: NeuralNet = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion: CrossEntropyLoss = CrossEntropyLoss()
optimizer: Adam = Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps: int = len(train_data_loader)
for epoch in range(num_epochs):  # type: int
    for i, (images, labels) in enumerate(train_data_loader):
        # 100, 1, 28, 28
        # 100, 784
        images: Tensor = images.reshape(-1, 28 * 28).to(device)
        labels: Tensor = labels.to(device)

        # forward
        outputs: Tensor = model(images)
        loss: Tensor = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch [{epoch + 1} / {num_epochs}], step [{i + 1} / {n_total_steps}], loss = {loss.item():.4f}')

# validation ?

# test
with torch.no_grad():
    n_correct: int = 0
    n_samples: int = 0
    for images, labels in test_data_loader:
        images: Tensor = images.reshape(-1, 28 * 28).to(device)
        labels: Tensor = labels.to(device)
        outputs: Tensor = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]  # type: int
        n_correct += (predictions == labels).sum().item()  # type: int

acc: float = 100. * n_correct / n_samples
print(f'accuracy = {acc}')
