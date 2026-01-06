import torch, torch.nn as nn
import torchvision.models as tvm


class SmallCNN(nn.Module):
  def __init__(self, in_ch=3, n_classes=2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_ch,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
      nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
      nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
    self.fc = nn.Linear(128, n_classes)
  
  def forward(self,x):
    feat=self.net(x).flatten(1)
    
    return self.fc(feat)
  
def make_resnet18(n_classes=2, in_ch=3, pretrained=True):
  m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

  if in_ch != 3:
    w = m.conv1.weight; m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)

    if in_ch == 1:
      with torch.no_grad(): m.conv1.weight.copy_(w.sum(dim=1, keepdim=True))
      
  in_dim = m.fc.in_features; m.fc = nn.Linear(in_dim, n_classes)
  
  return m
