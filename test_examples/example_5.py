import torch
import time


x = torch.rand([500, 10])
z = torch.zeros([10])

start = time.time()

for i in range(500):
    z += x[i]

print("Took {0:0.2e} seconds".format(time.time() - start))


z = torch.zeros([10])
start = time.time()

for x_i in torch.unbind(x):
    z += x_i
print("Unbind: Took {0:0.2e} seconds".format(time.time() - start))


start = time.time()
z = torch.sum(x, dim=0)
print("torch.sum: Took {0:0.2e} seconds".format(time.time() - start))
