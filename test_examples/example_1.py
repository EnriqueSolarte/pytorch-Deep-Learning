import torch
import numpy as np
from torch.nn.modules import loss

# * Parameters
w = torch.tensor(torch.randn([3, 1]), requires_grad=True)

# * Adam optimizer with learining rate 0.1
opt = torch.optim.Adam([w], 0.1)


def model(x):
    f = torch.stack([x*x, x, torch.ones_like(x)], 1)
    y_hat = torch.squeeze(f @ w, 1)
    return y_hat


def compute_loss(y, y_hat):
    loss = torch.nn.functional.mse_loss(y_hat, y)
    return loss


def generate_data():
    x = torch.rand(100) * 20 - 10
    y = 5 * x * x + 3
    return x, y


def train_step():

    x, y = generate_data()
    y_hat = model(x)

    loss = compute_loss(y=y, y_hat=y_hat)

    opt.zero_grad()
    loss.backward()

    opt.step()


for _ in range(10000):
    train_step()

print(w.detach().numpy())
