#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import logging
import math
import torch
import numpy  as np

from espnet.nets.pytorch_backend.nets_utils import pad_list
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def plot_alignment(alignment, path, title=None, split_title=False, max_len=None):
    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'

    if split_title:
        title = split_title_line(title)

    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()
    

def gaussian_func(x, mean, variance):
    return torch.exp(-torch.pow(x - mean, 2) / (2 *  torch.pow(variance,2))) / (variance * math.sqrt(2 * 3.141592653589793))



def get_mean(dur):
    dur=dur.float()
    
    dur_=torch.cat([torch.FloatTensor([0.]).to(dur.device),dur],dim=0)
    cumsum=torch.cumsum(dur_,dim=0)
    dur= torch.cat([dur,torch.FloatTensor([0.]).to(dur.device)], dim=0)
    mean=cumsum+dur/2
    return mean[:-1]


def gaussian_sample(dur,variance,ilen,olen):

    variance = variance[:ilen]
    dur = dur[:ilen]
    mean = get_mean(dur).unsqueeze(-1)
    variance = variance.unsqueeze(-1)

    index = torch.arange(1, olen + 1).unsqueeze(0).repeat(repeats=(dur.shape[0], 1)).float().to(dur.device)
    pro = gaussian_func(x=index, mean=mean, variance=variance)

    pro_norm = pro / torch.sum(pro, dim=0).unsqueeze(0)
    

    return pro_norm.transpose(1,0)



class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, max_frame=None,alpha=1.0):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            ilens (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        if max_frame is None:
            max_frame = torch.max(torch.sum(ds, dim=-1)).item()
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
            ds = torch.clamp(ds,1,50).long()
            max_frame=math.ceil(max_frame*alpha)
        xs = [x for x   in xs]
        ds = [d for d   in ds]


        xs = [self._repeat_one_sequence(x, d, max_frame) for x, d in zip(xs, ds)]
        xs = torch.stack(xs,dim=0)

        # xs= pad_list(xs, self.pad_value)
        # if max_frame is not None and max_frame-xs.shape[1]>0:
        # xs=torch.cat([xs,xs)],dim=1)

        return xs

    def _repeat_one_sequence(self, x, d, max_frame):
        """Repeat each frame according to duration.

        Examples:
            >>> x = torch.tensor([[1], [2], [3]])
            tensor([[1],
                    [2],
                    [3]])
            >>> d = torch.tensor([1, 2, 3])
            tensor([1, 2, 3])
            >>> self._repeat_one_sequence(x, d)
            tensor([[1],
                    [2],
                    [2],
                    [3],
                    [3],
                    [3]])

        """
        # if d.sum() == 0:
            # # logging.warning("all of the predicted durations are 0. fill 0 with 1.")
            # d = d.fill_(1)
        result = torch.cat(
            [x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0
        )
        if max_frame is not None:
            frame = result.shape[0]
            result = torch.cat([result, torch.zeros([max_frame - frame, result.shape[1]],device=result.device)],dim=0)

        return result
        
        
class LengthRegulator_gs(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value
        self.index=9901
        

 

    def forward(self,xs,dur,variance,ilen,olen,i_max,o_max):
        weights = []
        if  ilen is None and olen is None:
            ilen=dur.shape[1]
            olen=torch.sum(dur).long()
            g = gaussian_sample(dur.squeeze(0), variance.squeeze(0), ilen, olen)
            weight=g.cpu().numpy().T
            plot_alignment(weight,path=os.path.join('/data1/qiangchunyu/wsm/espnet/egs/csmsc/tts1/fig','%d_Gaussian.png'%self.index),
                            title =np.squeeze(variance.cpu().numpy()))
            
            
            
            self.index+=1
            weights.append(g)
        else:
            
            for d, v, i, o in zip(dur, variance, ilen, olen):
                g = gaussian_sample(d, v, i, o)
                g = F.pad(g, [0, i_max - i, 0, o_max - o])
                weights.append(g)
            

        weights = torch.stack(weights, dim=0)
        result = torch.matmul(weights, xs)
        return result


