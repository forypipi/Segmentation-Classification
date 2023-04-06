import torch

model = torch.load("/data/orfu/DeepLearning/Segmentation-Classification/deep learning/seg/model/nnUNet2D.pth")
print(model['network_weights'].keys())