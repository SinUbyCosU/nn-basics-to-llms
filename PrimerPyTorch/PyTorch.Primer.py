import torch
print("Exercise 1: Understanding Shapes")

images=torch.rand(3,3,4,4)
flat_images=image.view(3,-1)
print("Flattened Images Shape:",flat_images.shape)

unsqueezed= flat_images.unsqueeze(1)
print("Unsqueezed Shape:", unsqueezed.shape)

prmuted=images.permute(0,2,3,1)
print("Permuted Shape",prmuted.shape)
