import torch

x = torch.rand(2, 3)
y = torch.rand(2, 3)
print(x)
print(y)
z = x * y
print(z)

x = torch.rand(5, 3)
print(x[1, :])

x = torch.rand(4, 4)
y = x.view(-1, 8)
print(y)
print(y.size())

