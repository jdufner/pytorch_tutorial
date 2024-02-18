import torch
import torch.nn as nn
import numpy as np


# Numpy
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


Y = np.array([1, 0, 0])

y_pred_good = np.array([.7, .2, .1])
y_pred_bad = np.array([.1, .3, .6])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')


# PyTorch
loss = nn.CrossEntropyLoss()

# 1 sample
Y = torch.tensor([0])
Y_pred_good = torch.tensor([[2., 1., .1]])
Y_pred_bad = torch.tensor([[.5, 2., .3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)
print(prediction1)
print(prediction2)

# 3 samples
Y = torch.tensor([2, 0, 1])
Y_pred_good = torch.tensor([[.1, 1., 2.1], [2., 1., .1], [.1, 3., .1]])
Y_pred_bad = torch.tensor([[2.1, 1., .1], [.1, 1., 2.1], [.1, 3., .1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())

_, prediction1 = torch.max(Y_pred_good, 1)
_, prediction2 = torch.max(Y_pred_bad, 1)
print(prediction1)
print(prediction2)
