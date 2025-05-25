#In this file we took the DLG algorithm for testing data reconstruction
#for plain data, using gradient compression and using gradient noising.
#Only for classification CNNs.
#We followed the implementations from the original DLG paper https://arxiv.org/abs/1906.08935


import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms


dst = datasets.CIFAR100("tmp/files/torch", download=False)
datasets.CIFAR100
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def show_image_grid(images, loss):
    n_images = len(images)
    if n_images > 1:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()


        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        plt.imshow(images[0])
        plt.axis('off')
        plt.title(loss)
        plt.show()

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act()
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


net = LeNet().to(device)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

#Image from cifar10 to be reconstructed
img_index = [7]
gt_data = [tp(dst[i][0]).to(device) for i in img_index]
show_image_grid([dst[i][0] for i in img_index], 0)
gt_data = torch.stack(gt_data)
gt_label = torch.Tensor([dst[i][1] for i in img_index]).long().to(device)
gt_label = gt_label.view(len(img_index), )
gt_onehot_label = label_to_onehot(gt_label)

net = LeNet().to(device)


net.apply(weights_init)
criterion = cross_entropy_for_onehot


pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
grad = torch.autograd.grad(y, net.parameters())

original_grad = list((_.detach().clone() for _ in grad))
#optional gradient noising
#noise_std = 0.01
#original_grad = [grad + torch.randn_like(grad) * noise_std for grad in original_grad]


# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
print(dummy_data.shape)
print(dummy_label.shape)

show_image_grid([tt(dummy_data[i].cpu()) for i in range(len(img_index))],0)


#perform DLG reconstruction
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
history = []
for iters in range(1000000):
    def closure():
        optimizer.zero_grad()
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_grad = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_grad, original_grad):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        return grad_diff

    diff = optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append([tt(dummy_data[i].cpu()) for i in range(len(img_index))])
        #show the latest reconstructed version
        show_image_grid(history[-1], current_loss)