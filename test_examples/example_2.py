import torch


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # * basically these Parameters are tensor with requires_grad in true
        self.a = torch.nn.Parameter(torch.rand(1))
        self.b = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        y_hat = self.a * x + self.b
        return y_hat


x = torch.arange(100, dtype=torch.float32)

net = Net()

y = net(x)


for p in net.parameters():
    print(p)


x = torch.arange(100, dtype=torch.float32)/100
y = 5 * x + 3 + torch.rand(100)*0.3


criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for i in range(10000):
    net.zero_grad()
    y_hat = net(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()

print(net.a, net.b)





