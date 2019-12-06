"""
Coord Conv pytorch
"""

import torch
import torch.nn as nn


class AddCoords(nn.Module):
    """Add coords to a tensor"""
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        device = input_tensor.device
        batch_size_tensor = input_tensor.shape[0]  # get batch size

        xx_ones = torch.ones([batch_size_tensor, self.x_dim], dtype=torch.float, device=device, requires_grad=True)  # e.g. (batch, x)
        xx_ones = xx_ones.unsqueeze(-1)  # e.g. (batch, x, 1)

        xx_range = torch.arange(self.y_dim, dtype=torch.float, device=device, requires_grad=True).unsqueeze(0).repeat(batch_size_tensor, 1)  # e.g. (batch, y)
        xx_range = xx_range.unsqueeze(1)  # e.g. (batch, 1, y)

        xx_channel = torch.matmul(xx_ones, xx_range)  # e.g. (batch, x, y)
        xx_channel = xx_channel.unsqueeze(1)  # e.g. (batch, x, y, 1)

        yy_ones = torch.ones([batch_size_tensor, self.y_dim], dtype=torch.float, device=device, requires_grad=True)  # e.g. (batch, y)
        yy_ones = yy_ones.unsqueeze(1)  # e.g. (batch, 1, y)
        yy_range = torch.arange(self.x_dim, dtype=torch.float, device=device, requires_grad=True).unsqueeze(0).repeat(batch_size_tensor, 1)  # (batch, x)
        yy_range = yy_range.unsqueeze(-1)  # e.g. (batch, x, 1)

        yy_channel = torch.matmul(yy_range, yy_ones)  # e.g. (batch, x, y)
        yy_channel = yy_channel.unsqueeze(1)  # e.g. (batch, 1, x, y)

        xx_channel = xx_channel.float() / (self.y_dim - 1)
        yy_channel = yy_channel.float() / (self.x_dim - 1)
        xx_channel = xx_channel * 2 - 1  # [-1,1]
        yy_channel = yy_channel * 2 - 1

        ret = torch.cat([input_tensor,
                         xx_channel,
                         yy_channel], dim=1)  # e.g. (batch, c+2, x, y)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)  # e.g. (batch, c+3, x, y)

        return ret


class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, *args,  **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim,
                                   y_dim=y_dim,
                                   with_r=with_r)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret
