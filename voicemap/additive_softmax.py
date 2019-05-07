import torch
import torch.nn.functional as F
from torch import nn


class AdditiveSoftmaxLinear(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, scale: float = 30, margin: float = 0.35):
        super(AdditiveSoftmaxLinear, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.Tensor(input_dim, num_classes), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight)
        # def l2_normalise(module, input):
        #     return module.weight / (module.weight.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-8)
        # self.weight
        # self.weight.register_forward_pre_hook(l2_normalise)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None):
        if self.training:
            assert labels is not None
            # Normalise embeddings
            embeddings = embeddings.div((embeddings.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8))

            # Normalise weights
            normed_weights = self.weight / (self.weight.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-8)

            cos_theta = torch.mm(embeddings, normed_weights)
            phi = cos_theta - self.margin

            labels_onehot = F.one_hot(labels, self.num_classes).byte()
            logits = self.scale * torch.where(labels_onehot, phi, cos_theta)

            return logits
        else:
            assert labels is None
            return torch.mm(embeddings, self.weight)
