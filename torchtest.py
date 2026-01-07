import torch
print(f"Location: {torch.__file__}") 
print(f"Vulkan:   {torch.is_vulkan_available()}")
print(f"CUDA:     {torch.cuda.is_available()}")

# The ultimate proof: Can we move a tensor to the 'Fire' (GPU)?
x = torch.ones(1, device='cuda')
print(f"Tensor is on: {x.device}")