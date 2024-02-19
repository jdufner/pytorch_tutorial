import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 5
batch_size = 4
learning_rate = .001

# dataset has PILImage image or range [0, 1]
# transform images to tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# def imshow(img):
#     img = img / 2 + 0.5  # de-normalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(train_data_loader)
# images, labels = next(dataiter)
#
# # show images
# imshow(torchvision.utils.make_grid(images))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # padding = 0, stride = 1
        self.pool = nn.MaxPool2d(2, 2)  # padding = 0
        self.conv2 = nn.Conv2d(6, 16, 5)  # padding = 0, stride = 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # formula to calculate new image size (width - filter (aka kernel_size) + 2 * padding) / stride
        # (32 - 5 + 2 * 0) / 1 + 1 = 28 -> reduce image size from 32x32 to 28x28
        # (28 - 2 + 2 * 0) / 2 + 1 = 14 -> reduce image size from 28x28 to 14x14
        x = self.pool(F.relu(self.conv1(x)))
        # (14 - 5 + 2 * 0) / 1 + 1 = 10 -> reduce image size from 14x14 to 10x10
        # (10 - 2 + 2 * 0) / 2 + 1 = 5 -> reduce image size from 10x10 to 5x5
        x = self.pool(F.relu(self.conv2(x)))
        # Creates a 1d-tensor out of a 2d-tensor
        # Length of the 1d-tensor is number of channels (16) * dimension of image (5x5) -> 400
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        # origin shape: [4, 3, 32, 32 = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'epoch [{epoch + 1} / {num_epochs}], step [{i + 1} / {n_total_steps}], loss = {loss.item():.4f}')

print('Training finished')
# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100. * n_correct / n_samples
    print(f'accuracy of the network = {acc}')

    for i in range(10):
        acc = 100. * n_class_correct[i] / n_class_samples[i]
        print(f'accuracy of {classes[i]}: {acc} %')
