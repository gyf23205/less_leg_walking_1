import torch

B = 4
A = torch.randn(B, 9, requires_grad=True)

# Indices in the output that will be filled by A
out_indices = torch.tensor([0, 1, 3, 4, 5, 8, 9, 10, 11])

# Allocate output
Y = torch.zeros(B, 12, device=A.device, dtype=A.dtype)

# Fill non-zero positions
Y[:, out_indices] = A

# Test differentiability
loss = Y.sum()
loss.backward()

print(A.grad)  