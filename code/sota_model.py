"""IG65M-compatible R(2+1)D-34; no hub/network dependency at deployment.
Architecture corrections follow moabitcoin/ig65m-pytorch (MIT).
"""
from pathlib import Path
import torch
from torch import nn
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D

WEIGHTS=Path('/home/nyx/.cache/torch/hub/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth')

class Model(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        net=VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4,layers=[3,4,6,3],stem=R2Plus1dStem,num_classes=400)
        for layer,ch,mid in [(net.layer2,128,288),(net.layer3,256,576),(net.layer4,512,1152)]:
            layer[0].conv2[0]=Conv2Plus1D(ch,ch,mid)
        for m in net.modules():
            if isinstance(m,nn.BatchNorm3d): m.eps=1e-3;m.momentum=.9
        if pretrained: net.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
        old=net.stem[0]
        net.stem[0]=nn.Conv3d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        with torch.no_grad():
            net.stem[0].weight[:,:3].copy_(old.weight)
            net.stem[0].weight[:,3:].copy_(old.weight.mean(1,keepdim=True))
        features=net.fc.in_features;net.fc=nn.Identity()
        self.encoder=net
        self.head=nn.Sequential(nn.Dropout(.3),nn.Linear(features,40))

    def forward(self,x): return self.head(self.encoder(x))


def strideless(encoder, layers=('layer2', 'layer3', 'layer4')):
    """Remove temporal striding so every input frame keeps a feature position.

    Stride is not a learned parameter, so this changes the temporal resolution
    the encoder computes at without altering any weight. A checkpoint trained
    this way is byte-identical in shape to one trained with striding, which is
    exactly why inference must apply this explicitly rather than infer it.
    """
    changed = []
    for name in layers:
        for path, module in getattr(encoder, name).named_modules():
            if isinstance(module, nn.Conv3d) and module.stride[0] > 1:
                module.stride = (1,) + tuple(module.stride[1:])
                changed.append(f'{name}.{path}')
    return changed
