import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from typing import List


class GlobalMaxPool1d(nn.Module):
    def forward(self, input):
        return F.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool1d(nn.Module):
    def forward(self, input):
        return F.avg_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class Bottleneck(nn.Module):
    """Gets bottleneck features from an nn.Sequential classifier."""
    def __init__(self, model):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(*model[:-1])

    def forward(self, x):
        return self.bottleneck(x)


def get_classifier(filters, embedding, num_classes):
    return nn.Sequential(
        nn.Conv1d(1, filters, 32, padding=1),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=4, stride=4),

        nn.Conv1d(filters, 2 * filters, 3, padding=1),
        nn.BatchNorm1d(2 * filters),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(2 * filters, 3 * filters, 3, padding=1),
        nn.BatchNorm1d(3 * filters),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(3 * filters, 4 * filters, 3, padding=1),
        nn.BatchNorm1d(4 * filters),
        nn.ReLU(),

        GlobalAvgPool1d(),

        nn.Linear(4 * filters, embedding),

        nn.Linear(embedding, num_classes),
    )


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: int = None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2,)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if stride > 1:
            self.downsample = True
            self.downsample_op = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride),
                nn.ReLU()
            )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_op(x)

        out += identity
        out = F.relu(out)

        return out


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: int = None):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if stride > 1:
            self.downsample = True
            self.downsample_op = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride),
                nn.ReLU()
            )
        else:
            self.downsample = False

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_op(x)

        out += identity
        out = F.relu(out)

        return out


class ResidualClassifier(nn.Module):
    def __init__(self, filters: int, layers: List[int], num_classes: int):
        super(ResidualClassifier, self).__init__()
        self.filters = filters

        self.conv1 = nn.Conv1d(1, filters, kernel_size=15, stride=4, bias=False, padding=7)
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.maxpool1 = nn.MaxPool1d(kernel_size=9, stride=4)

        self.layer1 = self._make_layer(self.filters, filters, layers[0])
        self.layer2 = self._make_layer(filters, filters*2, layers[1], stride=4)
        self.layer3 = self._make_layer(filters*2, filters*4, layers[2], stride=4)
        self.layer4 = self._make_layer(filters*4, filters*8, layers[3], stride=4)
        self.avgpool = GlobalAvgPool1d()
        self.layer_norm = nn.LayerNorm(filters*8, elementwise_affine=False)

        self.fc = nn.Linear(filters*8, num_classes, bias=False)
        def l2_normalise(module, input):
            return module.weight / (module.weight.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-8)
        self.fc.register_forward_pre_hook(l2_normalise)

        # Rescaling lengthscale for loss
        self.l = 30

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))

        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, embed_only: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Normalise embeddings
        x = self.layer_norm(x)
        x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        x = x * self.l

        if embed_only:
            return x
        else:
            x = self.fc(x)
            return x