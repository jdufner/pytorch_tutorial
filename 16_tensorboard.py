# MNIST
# Dataloader, Transformation
# Multilayer Neural Net, Activation Function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# device config
device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set up writer for Tensorboard
writer = SummaryWriter("runs/MNIST2")

# hyper parameters
input_size = 784  # 28 x 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = .01

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_data_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28 * 28))
writer.close()

# training loop
n_total_steps = len(train_data_loader)

running_loss = 0.
running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'epoch [{epoch + 1} / {num_epochs}], step [{i + 1} / {n_total_steps}], loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.
            running_correct = 0

labels = []
preds = []

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels1 in test_data_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels1 = labels1.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels1.size(0)
        n_correct += (predictions == labels1).sum().item()

        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_predictions)
        labels.append(predictions)

    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)

acc = 100. * n_correct / n_samples
print(f'accuracy = {acc}')

classes = range(10)
for i in classes:
    labels_i = labels == i
    preds_i = preds[:, i]
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()
