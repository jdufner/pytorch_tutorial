# MNIST
# Dataloader, Transformation
# Multilayer Neural Net, Activation Function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Iterator, Tuple
import matplotlib as plt

# device config
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size: int = 784  # 28 x 28
hidden_size: int = 100
num_classes: int = 10
num_epochs: int = 2
batch_size: int = 100
learning_rate: float = .001

# MNIST
# Training data
train_dataset: torchvision.datasets = torchvision.datasets.MNIST(root='./data', train=True,
                                                                 transform=transforms.ToTensor(), download=True)
train_data_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Test data
test_dataset: torchvision.datasets = torchvision.datasets.MNIST(root='./data', train=False,
                                                                transform=transforms.ToTensor())
test_data_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# without type hints
# examples = iter(train_data_loader)
# samples, labels = next(examples)

# with type hints
examples: Iterator[Tuple[torch.Tensor, torch.Tensor]] = iter(train_data_loader)
example: Tuple[torch.Tensor, torch.Tensor] = next(examples)
samples: torch.Tensor = example[0]
labels: torch.Tensor = example[1]

print(samples.shape, labels.shape)

for i in range(6):  # type: int
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')


# plt.show()


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.relu: nn.ReLU = nn.ReLU()
        self.l2: nn.Linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> nn.Linear:
        out: nn.Linear = self.l1(x)
        out: nn.ReLU = self.relu(out)
        out: nn.Linear = self.l2(out)
        return out


model: NeuralNet = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps: int = len(train_data_loader)
for epoch in range(num_epochs):  # type: int
    for i, (images, labels) in enumerate(train_data_loader):
        # 100, 1, 28, 28
        # 100, 784
        images: torch.Tensor = images.reshape(-1, 28 * 28).to(device)
        labels: torch.Tensor = labels.to(device)

        # forward
        outputs: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch [{epoch + 1} / {num_epochs}], step [{i + 1} / {n_total_steps}], loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct: int = 0
    n_samples: int = 0
    for images, labels in test_data_loader:
        images: torch.Tensor = images.reshape(-1, 28 * 28).to(device)
        labels: torch.Tensor = labels.to(device)
        outputs: torch.Tensor = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]  # type: int
        n_correct += (predictions == labels).sum().item()  # type: int

acc: float = 100. * n_correct / n_samples
print(f'accuracy = {acc}')
