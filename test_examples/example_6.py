import torch

@torch.jit.script
def batch_gathe(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)
