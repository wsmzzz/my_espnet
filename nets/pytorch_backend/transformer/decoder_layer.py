#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, MultiHeadedAttention_v2
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward,FFN
import copy


class Add_norm(nn.Module):

    def __init__(self, num_hidden):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)
        

    def forward(self, inputs, memory, query_mask):
        qm = query_mask.unsqueeze(-1).repeat(1, 1, inputs.shape[-1]).float()
        frame=inputs.shape[1]
        return self.layer_norm(inputs + self.dropout(memory [:,-frame:,:]* qm))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class fft_block_add(nn.Module):
    def __init__(self, idim, hidden_units, ffn_dropout_rate, n_head):
        super().__init__()
        self.selfattn_layers = MultiHeadedAttention_v2(n_head, idim)
        self.layer_norm_1 = nn.LayerNorm(idim)
        self.layer_norm_2 = nn.LayerNorm(idim)
        self.ffns = FFN(idim)
        self.adds = Add_norm(idim)

    def forward(self, query ,memory, attn_mask, query_mask,residual=False):
    

        outputs, attns = self.selfattn_layers(query, query, query, attn_mask, query_mask,residual=residual)
        outputs = self.layer_norm_1(outputs)
        outputs = self.adds(outputs, memory, query_mask)
        outputs = self.ffns(outputs, query_mask)
        outputs = self.layer_norm_2(outputs)

        return outputs, attns
    
    def forward_part(self, query , memory, attn_mask, query_mask,cache=None,residual=False):
        key=query
        query=query[:,-50:,:]
        outputs, attns = self.selfattn_layers(query, key, key, attn_mask, query_mask,residual=residual)
        outputs = self.layer_norm_1(outputs)
        outputs = self.adds(outputs, memory, query_mask)

        outputs = self.ffns(outputs, query_mask)
        outputs = self.layer_norm_2(outputs)
        if cache is None:
            outputs=outputs
        else:
            outputs=torch.cat([cache,outputs],dim=1)
        
        return outputs,None
        
    


class fft_block(nn.Module):

    def __init__(self, idim, hidden_units, ffn_dropout_rate, n_head):
        super().__init__()
        self.selfattn_layers = MultiHeadedAttention_v2(n_head, idim)
        self.ffns = FFN(idim)
        self.layer_norm_1 = nn.LayerNorm(idim)
        self.layer_norm_2 = nn.LayerNorm(idim)

    def forward(self, query , attn_mask, query_mask):
       
        outputs, attns = self.selfattn_layers(query, query, query, attn_mask, query_mask)
        outputs = self.layer_norm_1(outputs)
        outputs = self.ffns(outputs, query_mask)
        outputs = self.layer_norm_2(outputs)
       

        return outputs, attns

    def forward_part(self, query, attn_mask, query_mask,cache=None):
        key=query
        query=query[:,-50:,:]
        outputs, attns = self.selfattn_layers(query, key, key, attn_mask, query_mask)
        outputs = self.layer_norm_1(outputs)
        outputs = self.ffns(outputs, query_mask)
        outputs = self.layer_norm_2(outputs)
        if cache is None:
            outputs=outputs
        else:
            outputs=torch.cat([cache,outputs],dim=1)
        
        return outputs,None
        
class fft_block_plus(nn.Module):

    def __init__(self, idim, hidden_units, ffn_dropout_rate, n_head):
        super().__init__()
        self.fft_b_1=fft_block(idim, hidden_units, ffn_dropout_rate, n_head)
        self.fft_ab_1=fft_block_add( idim, hidden_units, ffn_dropout_rate, n_head)
        self.fft_b_2=fft_block(idim, hidden_units, ffn_dropout_rate, n_head)
        self.fft_ab_2=fft_block_add(idim, hidden_units, ffn_dropout_rate, n_head)
 

    def forward(self, query , memory, attn_mask, query_mask,cache=None):
        x=query
      
        x,_=self.fft_b_1(query=x , attn_mask=attn_mask, query_mask=query_mask)
        x,_=self.fft_ab_1(query=x , attn_mask=attn_mask, query_mask=query_mask,memory=memory,residual=True)
        x,_=self.fft_b_2(query=x , attn_mask=attn_mask, query_mask=query_mask)
        x,_=self.fft_ab_2(query=x , attn_mask=attn_mask, query_mask=query_mask,memory=memory,residual=True)
        return x
        
        

class Phone_Autoaggressive_DecoderLayer_v2(nn.Module):
    """Single decoder layer module.

        :param int size: input dim
        :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
            self_attn: self attention module
        :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
            src_attn: source attention module
        :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
            PositionwiseFeedForward feed_forward: feed forward layer module
        :param float dropout_rate: dropout rate
        :param bool normalize_before: whether to use layer_norm before the first block
        :param bool concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

        """

    def __init__(
            self,
            size,
            hidden_units,
            ffn_dropout_rate,
            n_head,
            normalize_before=True
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.heads = n_head
        self.pre_fft_block = fft_block_add(idim=size,
                                           hidden_units=hidden_units,
                                           ffn_dropout_rate=ffn_dropout_rate,
                                           n_head=n_head)
        self.fft_block = fft_block_plus(idim=size,
                                          hidden_units=hidden_units,
                                          ffn_dropout_rate=ffn_dropout_rate,
                                          n_head=n_head)

        self.norm = LayerNorm(size)
        self.normalize_before = normalize_before
        
    def forward(self, query, memory, pad_mask, dur_mask, query_mask,decoder_cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        
        if decoder_cache is None:
        
            new_cache=None
            dur_mask = dur_mask.eq(0).repeat(self.heads, 1, 1)
            pad_mask = pad_mask.eq(0).repeat(self.heads, 1, 1)

            if self.normalize_before:
                query = self.norm(query)
            attn_dec_list = list()

            output, attns = self.pre_fft_block(query, memory, pad_mask, query_mask)
            output = self.fft_block(query = output, 
                                    memory = memory, 
                                    attn_mask = dur_mask, 
                                    query_mask =query_mask)
           
        else :
           raise RuntimeError('not write')
            
        return output, attn_dec_list,new_cache
        
    


 
        
    

class Phone_Autoaggressive_DecoderLayer(nn.Module):
    """Single decoder layer module.

        :param int size: input dim
        :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
            self_attn: self attention module
        :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
            src_attn: source attention module
        :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
            PositionwiseFeedForward feed_forward: feed forward layer module
        :param float dropout_rate: dropout rate
        :param bool normalize_before: whether to use layer_norm before the first block
        :param bool concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

        """

    def __init__(
            self,
            size,
            hidden_units,
            ffn_dropout_rate,
            n_head,
            normalize_before=True
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.heads = n_head
        self.pre_fft_block = fft_block_add(idim=size,
                                           hidden_units=hidden_units,
                                           ffn_dropout_rate=ffn_dropout_rate,
                                           n_head=n_head)
        self.fft_block = clones(fft_block(idim=size,
                                          hidden_units=hidden_units,
                                          ffn_dropout_rate=ffn_dropout_rate,
                                          n_head=n_head)
                                , 5)
        self.decoder_num=1+len(self.fft_block)
        self.norm = LayerNorm(size)
        self.normalize_before = normalize_before
        
    def forward(self, query, memory, pad_mask, dur_mask, query_mask,decoder_cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        
        if decoder_cache is None:
        
            new_cache=None
            dur_mask = dur_mask.eq(0).repeat(self.heads, 1, 1)
            pad_mask = pad_mask.eq(0).repeat(self.heads, 1, 1)

            if self.normalize_before:
                query = self.norm(query)
            attn_dec_list = list()
            output, attns = self.pre_fft_block(query, memory, pad_mask, query_mask)
            attn_dec_list.append(attns)
            for block in self.fft_block:
                output,attns = block(output,dur_mask, query_mask)
                attn_dec_list.append(attns)
        else :
            new_cache=[]
            dur_mask = dur_mask.eq(0).repeat(self.heads, 1, 1)
            pad_mask = pad_mask.eq(0).repeat(self.heads, 1, 1)
            if self.normalize_before:
                query = self.norm(query)
            attn_dec_list = list()
            output, attns = self.pre_fft_block.forward_part(query, memory, pad_mask, query_mask,decoder_cache[0])
            new_cache.append(output)
            attn_dec_list.append(attns)
            for block,cache in zip(self.fft_block,decoder_cache[1:-1]):
                output,attns = block.forward_part(output,dur_mask, query_mask,cache)
                attn_dec_list.append(attns)
                new_cache.append(output)
            
        return output, attn_dec_list,new_cache

    


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward: feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
            self,
            size,
            self_attn,
            src_attn,
            feed_forward,
            dropout_rate,
            normalize_before=True,
            concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask
