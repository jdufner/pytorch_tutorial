import torch
# import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))

# a.add_(1)
a += 1
print(a)
print(b)

x = torch.ones(5, requires_grad=True)
print(x)
