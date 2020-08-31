import torch
import torch.nn as nn
import torch.nn.functional as F

from net import resnet50

def globalaveragepooling2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), dim=-1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    
    return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        
        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)
        
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        
        self.classifier = nn.Conv2d(in_channels=2048, out_channels=20, kernel_size=1, bias=False)
    def forward(self, x):
        N, C, H, W = x.size()
        print("N, C, H, W", N, C, H, W)
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()
        
        #out = globalaveragepooling2d(x5, keepdims=True)
        out = self.classifier(x5)
        cam = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
        #out = out.view(-1, 20)
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        #edge_out = torch.sigmoid(edge_out)
        print('x5', x5.shape)
        print('out', out.shape)
        print('cam', cam.shape)
        print('edge)out', edge_out.shape)
        return edge_out
    
    def trainable_parameters(self):
        return tuple(self.edge_layers.parameters())
    
    
class EDM(Net):
    def __init__(self, crop_size=512, stride=4):
        super(EDM, self).__init__()
        self.crop_size = crop_size
        self.stride = stride
        
    def forward(self, x):
        feature_size = (x.size(2)-1) // self.stride+1, (x.size(3)-1) // self.stride+1
        
        x = F.pad(x, [0, self.crop_size-x.size(3), 0, self.crop_size-x.size(2)])
        edge_out = super().forward(x)
        edge_out = edge_out[..., :feature_size[0], :feature_size[1]]
        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        
        return edge_out
