import torch
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2., 1., .1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2., 1., .1])
outputs = torch.softmax(x, dim=0)
print('softmax torch', outputs)
