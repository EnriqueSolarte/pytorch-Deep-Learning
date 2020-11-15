import torch


a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1], [2]])

c = a + b.repeat([1, 2])

print(c)


a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])

linear = torch.nn.Linear(11, 10)

tiled_b = b.repeat([1, 3, 1])

c = torch.cat([a, tiled_b], 2)

d = torch.nn.functional.relu(linear(c))

print(d.shape)

# ! But this can be done more effecient by

linear1 = torch.nn.Linear(5, 10)
linear2 = torch.nn.Linear(6, 10)

pa = linear1(a)
pb = linear2(b)

d2 = torch.nn.functional.relu(pa + pb)

print(d2.shape)

print("done")