import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from typing import List, Callable


from voicemap.additive_softmax import AdditiveSoftmaxLinear


class GlobalMaxPool1d(nn.Module):
    def forward(self, input):
        return F.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool1d(nn.Module):
    def forward(self, input):
        return F.avg_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ConvBlock, self).__init__()
        self.conv_type = nn.Conv1d if dim == 1 else nn.Conv2d
        self.bn_type = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d

        self.conv = self.conv_type(in_channels, out_channels, kernel_size, padding=1)
        self.bn = self.bn_type(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BaselineClassifier(nn.Module):
    def __init__(self, in_channels, filters, embedding, num_classes, dim: int = 1):
        super(BaselineClassifier, self).__init__()
        self.dim = dim

        self.conv1 = ConvBlock(in_channels, filters, 3, self.dim)
        self.conv2 = ConvBlock(filters, 2*filters, 3, self.dim)
        self.conv3 = ConvBlock(2*filters, 3*filters, 3, self.dim)
        self.conv4 = ConvBlock(3*filters, 4*filters, 3, self.dim)
        self.max_pool = F.max_pool1d if dim == 1 else F.max_pool2d
        self.avg_pool = GlobalAvgPool1d()
        if self.dim == 2:
            self.finalconv = nn.Conv1d(4*filters, 4*filters, kernel_size=(10, 1))
        self.embedding = nn.Linear(4*filters, embedding, bias=False)
        self.fc = AdditiveSoftmaxLinear(embedding, num_classes)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None,
                return_embedding: bool = False):
        x = self.max_pool(self.conv1(x), kernel_size=2, stride=2)
        x = self.max_pool(self.conv2(x), kernel_size=2, stride=2)
        x = self.max_pool(self.conv3(x), kernel_size=2, stride=2)
        x = self.max_pool(self.conv4(x), kernel_size=2, stride=2)
        if self.dim == 2:
            x = self.finalconv(x)
            x = x.squeeze()

        x = self.avg_pool(x)

        x = self.embedding(x)
        if return_embedding:
            return x
        else:
            x = self.fc(x, y)
            return x


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: int = None,
                 conv_type: Callable = nn.Conv1d):
        super(ResidualBlock1D, self).__init__()
        self.conv_type = conv_type
        self.conv1 = self.conv_type(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = self.conv_type(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if stride > 1:
            self.downsample = True
            self.downsample_op = nn.Sequential(
                self.conv_type(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
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
    def __init__(self, in_channels: int, filters: int, layers: List[int], num_classes: int, dim: int = 1):
        super(ResidualClassifier, self).__init__()
        self.filters = filters
        self.dim = dim
        self.conv_type = nn.Conv1d if dim == 1 else nn.Conv2d

        self.conv1 = self.conv_type(in_channels, filters, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.filters)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.filters, filters, layers[0])
        self.layer2 = self._make_layer(filters, filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(filters*2, filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(filters*4, filters*8, layers[3], stride=2)
        self.avgpool = GlobalAvgPool1d()
        self.layer_norm = nn.LayerNorm(filters*8, elementwise_affine=False)

        self.fc = AdditiveSoftmaxLinear(filters*8, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, conv_type=self.conv_type))

        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, conv_type=self.conv_type))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, return_embedding: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Normalise embeddings
        x = self.layer_norm(x)

        if return_embedding:
            return x
        else:
            x = self.fc(x, y)
            return x
