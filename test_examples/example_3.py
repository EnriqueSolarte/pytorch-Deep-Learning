import torch


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # * basically these Parameters are tensor with requires_grad in true
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_hat = self.linear(x.unsqueeze(1)).squeeze(1)
        return y_hat


x = torch.arange(100, dtype=torch.float32)

net = Net()

y = net(x)


for p in net.parameters():
    print(p)
