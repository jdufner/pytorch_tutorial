import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x.mean()
print(y)

y.backward()
print(x.grad)

# x.requires_grad_(False)
v = x.detach()
print(v)


weights = torch.ones(4, requires_grad=True)
b = weights.sum().backward()
weights.grad.zero_()
