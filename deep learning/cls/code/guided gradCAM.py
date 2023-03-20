import torch
import torch.nn as nn
import numpy as np
import cv2
class GuidedBackpropReLU(nn.Module):
    def __init__(self):
        super(GuidedBackpropReLU, self).__init__()
        self.relu = nn.ReLU()
        self.gradient = None
    def forward(self, x):
        self.gradient = x.grad
        return torch.clamp(x, min=0.0)
class GuidedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradient_values = None
        self.activations = None
        self.register_hooks()
    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient_values = grad_output[0].cpu().data.numpy()
        def forward_hook(module, input, output):
            self.activations = output.cpu().data.numpy()
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_layer = module
                break
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    def generate_gradients(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        one_hot_output = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot_output[0][target_class] = 1
        one_hot_output = torch.from_numpy(one_hot_output).requires_grad_(True)
        one_hot_output = torch.sum(one_hot_output * output)
        one_hot_output.backward(retain_graph=True)
        gradients = self.gradient_values[0]
        activations = self.activations[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros((activations.shape[1], activations.shape[2]), dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.float32(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET))
        guided_gradients = np.zeros_like(input_image.cpu().numpy())
        input_image = input_image.cpu().numpy()[0].transpose((1, 2, 0))
        for i in range(gradients.shape[1]):
            filter_weights = weights[i]
            if filter_weights > 0:
                filter_gradients = gradients[i, :, :]
                filter_gradients = cv2.resize(filter_gradients, input_image.shape[:2])
                filter_gradients = np.float32(filter_gradients) / np.max(filter_gradients)
                filter_cam = cam[:, :, i]
                guided_gradients += filter_gradients * np.expand_dims(filter_cam, axis=2)
        guided_gradients = np.transpose(guided_gradients, (2, 0, 1))
        return guided_gradients
    
# 使用示例
# 假设我们有一个训练好的模型model和一个目标层target_layer
# 并且我们想可视化输入图像input_image的类别target_class的Grad-CAM和Guided Grad-CAM
gradcam = GradCAM(model, target_layer)
guided_gradcam = GuidedGradCAM(model, target_layer)
output = model(input_image)
predicted_class = torch.argmax(output).item()
gradcam_map = gradcam.generate_cam(input_image, predicted_class)
guided_gradcam_map = guided_gradcam.generate_gradients(input_image, predicted_class)
# 可视化Grad-CAM和Guided Grad-CAM