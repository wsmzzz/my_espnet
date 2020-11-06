#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x,query_mask=None):
        """Forward funciton."""
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]).float()
            return self.w_2(self.dropout(torch.relu(self.w_1(x))))*query_mask
        else:
            return self.w_2(self.dropout(torch.relu(self.w_1(x))))
            
class Conv(torch.nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x

            
            
            
class FFN(torch.nn.Module):
    """
    Positionwise Feed-Forward Network
    """
    
    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        self.w_1 = torch.nn.Linear(num_hidden, num_hidden * 2)
        self.w_2 = torch.nn.Linear(num_hidden * 2, num_hidden)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.layer_norm = torch.nn.LayerNorm(num_hidden)

    def forward(self, input_,mask):
        # FFN Network
        x=self.w_2(self.dropout(torch.relu(self.w_1(input_))))
        # residual connection
        x = x + input_
        x = self.layer_norm(x)
        q_mask=mask.unsqueeze(2).repeat([1,1,x.shape[2]]).float()
        x=x*q_mask
        return x
