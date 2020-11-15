import torch
import config
from file_utilities import *

x = torch.tensor(1.0, requires_grad=True)

var = torch.tensor((1, 2, 3, 4, 5, 6))


def u(x):
    return x * x


def g(u):
    return -u


dgdx = torch.autograd.grad(g(u(x)), x)[0]
print(dgdx)
