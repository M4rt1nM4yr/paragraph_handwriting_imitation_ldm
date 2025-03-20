import numpy as np
import torch

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = ~(torch.from_numpy(subsequent_mask) == 0).squeeze(0)
    matrix_ninf = torch.ones(()) * float('-inf')
    matrix_zeros = torch.zeros(()).float()
    subsequent_mask = torch.where(subsequent_mask,matrix_ninf,matrix_zeros)
    return subsequent_mask

if __name__ == "__main__":
    t = subsequent_mask(5)
    print(t)