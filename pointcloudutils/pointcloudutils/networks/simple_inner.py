"""
Adapted from https://github.com/FlyingGiraffe/vnn-neural-implicits/
LICENSE: https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/LICENSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cyto_dl.nn.point_cloud.vnn import VNLinear


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class DecoderInner(nn.Module):
    """Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=128, leaky=False):
        super().__init__()
        self.c_dim = c_dim

        if c_dim > 0:
            self.c_in = VNLinear(c_dim, c_dim)

        self.fc_in = nn.Linear(c_dim * 2 + 1, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p_, c_, **kwargs):
        batch_size, T, D = p_.size()

        if isinstance(c_, tuple):
            c_, c_meta = c_

        net = (p_ * p_).sum(2, keepdim=True)
        if self.c_dim != 0:
            c_ = c_.view(batch_size, -1, D).contiguous()
            net_c = torch.einsum("bmi,bni->bmn", p_, c_)
            c_dir = self.c_in(c_)
            c_inv = (c_ * c_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_c, c_inv], dim=2)

        net = self.fc_in(net)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
