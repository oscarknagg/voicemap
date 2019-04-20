from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


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

        GlobalMaxPool1d(),

        nn.Linear(4 * filters, embedding),

        nn.Linear(embedding, num_classes),
    )


def get_omniglot_classifier(num_classes, num_input_channels=1):
    return nn.Sequential(
        nn.Conv2d(num_input_channels, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        GlobalAvgPool2d(),

        nn.Linear(64, num_classes)
    )


def get_few_shot_encoder(num_input_channels=1):
    return nn.Sequential(
        nn.Conv2d(num_input_channels, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        Flatten(),
    )
