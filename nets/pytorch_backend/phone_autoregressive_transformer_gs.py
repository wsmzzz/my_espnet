#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Add gaussian sampling

"""TTS-Transformer related modules."""

import logging

import torch
import torch.nn.functional as F
import numpy as np
from time import time
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.pytorch_backend.transformer.decoder_layer import fft_block
from espnet.nets.pytorch_backend.e2e_asr_transformer import subsequent_mask
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.calculate_dtw import *
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.encoder import Rhythm_Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Phone_Autoaggressive_Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder_v2
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator_gs 
from espnet.nets.pytorch_backend.transformer.plot import _plot_and_save_attention
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import (
    DurationCalculator,  # noqa: H301
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import  Duration_variance_Predictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
DurationPredictorLoss)  # noqa: H301

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return  [total_num,trainable_num]


def get_phone_mask(dur,max_frame=None):


    Batch,Tx=dur.shape
    cusum_dur=np.cumsum(dur,axis=1)   #(N,T)
    if max_frame is None:
        max_frame=np.max(cusum_dur)
    masks=[]
    for i in range(Batch):
        mask=[]

        for j in range(Tx):
            for m in range(dur[i,j]):
                mask.append(np.pad(np.arange(1,cusum_dur[i,j]+1),(0,max_frame-cusum_dur[i,j]), mode='constant', constant_values=0))
        mask=np.stack(mask,axis=0)
        masks.append(mask)
    masks=[np.pad(mask,[(0,max_frame-cusum_dur[i,-1]),(0,0)]) for i, mask in enumerate(masks)]
    masks=np.stack(masks,axis=0)

    return  masks

def torch_get_phone_mask(dur,max_frame=None):

    Batch=dur.shape[0]
    Tx=dur.shape[1]
    cusum_dur=torch.cumsum(dur,dim=1)
    masks = []
    if max_frame is  None:
        max_frame=torch.max(cusum_dur)
    for i in range(Batch):
        mask=[]

        for j in range(Tx):
            for m in range(dur[i,j]):
                mask.append(F.pad(torch.arange(1,cusum_dur[i,j]+1,device=dur.device),(0,max_frame-cusum_dur[i,j]), mode='constant', value=0))
        mask=torch.stack(mask,dim=0)
        masks.append(mask)
    masks=[F.pad(mask,[0,0,0,max_frame-cusum_dur[i,-1]]) for i, mask in enumerate(masks)]
    masks=torch.stack(masks,dim=0)

    return  masks
    
def get_pad_mask(dur,pad_num,is_training=True):
    Batch, Tx = dur.shape
    dur_pad=np.concatenate([np.ones([Batch,1])*pad_num,dur[:,:-1]],axis=1).astype(np.int32)
    cusum_dur_real=np.cumsum(dur, axis=1)
    cusum_dur = np.cumsum(dur_pad, axis=1)  # (N,T)
    if is_training:
        max_frame = np.max(cusum_dur_real)
    else:
        max_frame=max(np.max(cusum_dur_real),np.max(cusum_dur))
    masks = []
    for i in range(Batch):
        mask = []

        for j in range(Tx):
            for m in range(dur[i, j]):
                if j==0:
                    mask.append(np.pad(np.arange(1, cusum_dur[i, j] + 1), (0, max_frame - cusum_dur[i, j]), mode='constant',
                                   constant_values=0))
                else:
                    mask.append(np.concatenate([np.zeros([pad_num]),
                                                np.arange(1, cusum_dur[i, j] + 1-pad_num),
                                                np.zeros([max_frame - cusum_dur[i, j]])],axis=0))



        mask = np.stack(mask, axis=0)

        masks.append(mask)
    masks = [np.pad(mask, [(0, max_frame - cusum_dur_real[i, -1]), (0, 0)]) for i, mask in enumerate(masks)]
    masks = np.stack(masks, axis=0)

    return masks

def torch_get_pad_mask(dur,pad_num,is_training=True):
    Batch = dur.shape[0]
    Tx = dur.shape[1]
    dur_pad = torch.cat([torch.ones([Batch,1],device=dur.device).long()*pad_num,dur[:,:-1]],dim=1)
    cusum_dur_real = torch.cumsum(dur, dim=1)
    cusum_dur = torch.cumsum(dur_pad, dim=1)  # (N,T)
    if is_training:
        max_frame = torch.max(cusum_dur_real)
    else:
        max_frame = torch.max(torch.max(cusum_dur_real), torch.max(cusum_dur))
    masks = []
    for i in range(Batch):
        mask = []

        for j in range(Tx):
            for m in range(dur[i, j]):
                if j == 0:
                    mask.append(
                        F.pad(torch.arange(1, cusum_dur[i, j] + 1,device=dur.device), (0, max_frame - cusum_dur[i, j]), mode='constant',
                               value=0))
                else:
                    mask.append(torch.cat([torch.zeros([pad_num],device=dur.device).long(),
                                                torch.arange(1, cusum_dur[i, j] + 1 - pad_num,device=dur.device),
                                                torch.zeros([max_frame - cusum_dur[i, j]],device=dur.device).long()], dim=0))

        mask = torch.stack(mask, dim=0)

        masks.append(mask)
    masks = [F.pad(mask, [0, 0,0, max_frame - cusum_dur_real[i, -1]]) for i, mask in enumerate(masks)]
    masks = torch.stack(masks, dim=0)

    return masks
    
def part_pad_mask(pre_dur,dur,max_frame):
    mask=[]
    for i in range(dur):
        mask.append(np.pad(np.arange(1,1+pre_dur),(0,max_frame-pre_dur)))
    mask=np.stack(mask,axis=0)
    mask=np.pad(mask,[[0,max_frame-dur],[0,0]])
    mask=np.expand_dims(mask,0)
    return  torch.from_numpy(mask).long()


class FeedForwardTransformerLoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False,stage=1):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.

        """
        super(FeedForwardTransformerLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.stage=stage
        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(self, after_outs, before_outs, d_outs, ys, ds, ilens, olens,stage=1):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_masks=torch.cat([duration_masks,duration_masks.new_zeros(duration_masks.shape[0],1)],dim=-1)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            if self.stage==1:

                out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
                buffer_size=ys.shape[1]-out_masks.shape[1]
                out_masks=torch.cat([out_masks,torch.zeros_like(out_masks[:,:buffer_size,:])],dim=1)
                before_outs = before_outs.masked_select(out_masks)
                after_outs = (
                    after_outs.masked_select(out_masks) if after_outs is not None else None
                )
                ys = ys.masked_select(out_masks)
                
                # calculate loss
                l1_loss = self.l1_criterion(before_outs, ys)
                if after_outs is not None:
                    l1_loss += self.l1_criterion(after_outs, ys)
            else:
                d_outs_int=xs = torch.clamp(
                    torch.round(d_outs_.exp() - 1), min=0
                    ).long()  # avoid negative value
                
                oplens=torch.sum(d_outs_int,dim=-1)  
                
                index_a, index_b = multi_process_path(ys,before_outs,olens,oplens)
                l1_loss=dtw_loss(ys,before_outs,index_a,index_b,self.l1_criterion)   
                if after_outs is not None:
                    l1_loss += dtw_loss(ys,after_outs,index_a,index_b,self.l1_criterion)
                
                
        duration_loss = self.duration_criterion(d_outs, ds)
        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )

        return l1_loss, duration_loss


class TransformerLoss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False, bce_pos_weight=20.0
    ):
        """Initialize Tactoron2 loss module.

        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token.

        """
        super().__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

        # NOTE(kan-bayashi): register pre hook function for the compatibility
        # self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, after_outs, before_outs, ys, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            buffer_size=ys.shape[1]-masks.shape[1]
            masks=torch.cat([masks,torch.zeros_like(masks[:,:buffer_size,:])],dim=1)
            
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
        )

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()

        return l1_loss, mse_loss

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Apply pre hook fucntion before loading state dict.

        From v.0.6.1 `bce_criterion.pos_weight` param is registered as a parameter but
        old models do not include it and as a result, it causes missing key error when
        loading old model parameter. This function solve the issue by adding param in
        state dict before loading as a pre hook function
        of the `load_state_dict` method.

        """
        key = prefix + "bce_criterion.pos_weight"
        # if key not in state_dict:
            # state_dict[key] = self.bce_criterion.pos_weight


class TTSPlot(PlotAttentionReport):
    """Attention plot module for TTS-Transformer."""

    def plotfn(self, data, attn_dict, outdir, suffix="png", savefn=None):
        """Plot multi head attentions.

        Args:
            data (dict): Utts info from json file.
            attn_dict (dict): Multi head attention dict.
                Values should be numpy.ndarray (H, L, T)
            outdir (str): Directory name to save figures.
            suffix (str): Filename suffix including image type (e.g., png).
            savefn (function): Function to save figures.

        """
        import matplotlib.pyplot as plt

        for name, att_ws in attn_dict.items():
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.%s.%s" % (outdir, data[idx][0], name, suffix)
                if "fbank" in name:
                    fig = plt.Figure()
                    ax = fig.subplots(1, 1)
                    ax.imshow(att_w, aspect="auto")
                    ax.set_xlabel("frames")
                    ax.set_ylabel("fbank coeff")
                    fig.tight_layout()
                else:
                    fig = _plot_and_save_attention(att_w, filename)
                savefn(fig, filename)
                
class Rhythm_Embedding(torch.nn.Module):
    
    def __init__(self,indim,outdim,rhythm_num,embed_rhythm_dim,concat_dim,padding=0,use_rhythm_embed=True):
        super().__init__()
        self.use_rhythm_embed=use_rhythm_embed
        self.embed = torch.nn.Embedding(indim, outdim, padding_idx=padding)
        if self.use_rhythm_embed:
            self.rhythm_embed=torch.nn.Embedding(rhythm_num,embed_rhythm_dim,padding_idx=padding)
            self.concat=torch.nn.Linear(outdim+embed_rhythm_dim,concat_dim)
    
    def forward(self,xs):
    
        if self.use_rhythm_embed:
            word_output = self.embed(xs[:,0,:].view(-1,xs.shape[-1]))
            rythm_output= self.rhythm_embed(xs[:,1,:].view(-1,xs.shape[-1]))
            enc_output=torch.cat([word_output,rythm_output],-1)
            enc_output=self.concat(enc_output)
        else:
            enc_output = self.embed(xs[:,0,:].view(-1,xs.shape[-1]))
        
        return enc_output




class Transformer(TTSInterface, torch.nn.Module):
    """Text-to-Speech Transformer module.

    This is a module of text-to-speech Transformer described
    in `Neural Speech Synthesis with Transformer Network`_,
    which convert the sequence of characters
    or phonemes into the sequence of Mel-filterbanks.

    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    """

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("transformer model setting")
        # network structure related
        group.add_argument(
            "--dur-frame-buffer",
            default=50,
            type=int,
            help="the length of duration buffer",
        )
        group.add_argument(
            "--inference-global",
            default=True,
            type=strtobool,
            help="the type of inference",
        )
        group.add_argument(
            "--embed-dim",
            default=512,
            type=int,
            help="Dimension of character embedding in encoder prenet",
        )
        group.add_argument(
            "--rhythm-num",
            default=5,
            type=int,
            help="Number of dimension of embedding",
        )
        group.add_argument(
        '--train-stage',
        default=1,
        type=int,
        )
        group.add_argument(
            "--embed-rhythm-dim",
            default=64,
            type=int,
            help="Number of dimension of embedding",
        )
        group.add_argument(
            "--dur-self-attn",
            default=False,
            type=strtobool,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--concat-dim",
            default=512,
            type=int,
            help="Number of dimension of embedding",
        )
        group.add_argument(
            "--dur-classfication",
            default=False,
            type=strtobool,
            help="Number of dimension of embedding",
        )
        group.add_argument(
            "--eprenet-conv-layers",
            default=3,
            type=int,
            help="Number of encoder prenet convolution layers",
        )
        group.add_argument(
            "--eprenet-conv-chans",
            default=256,
            type=int,
            help="Number of encoder prenet convolution channels",
        )
        group.add_argument(
            "--eprenet-conv-filts",
            default=5,
            type=int,
            help="Filter size of encoder prenet convolution",
        )
        group.add_argument(
            "--dprenet-layers",
            default=2,
            type=int,
            help="Number of decoder prenet layers",
        )
        group.add_argument(
            "--dprenet-units",
            default=256,
            type=int,
            help="Number of decoder prenet hidden units",
        )
        group.add_argument(
            "--elayers", default=3, type=int, help="Number of encoder layers"
        )
        group.add_argument(
            "--eunits", default=1536, type=int, help="Number of encoder hidden units"
        )
        group.add_argument(
            "--adim",
            default=384,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        group.add_argument(
            "--dlayers", default=3, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=1536, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--positionwise-layer-type",
            default="linear",
            type=str,
            choices=["linear", "conv1d", "conv1d-linear"],
            help="Positionwise layer type.",
        )
        group.add_argument(
            "--positionwise-conv-kernel-size",
            default=1,
            type=int,
            help="Kernel size of positionwise conv1d layer",
        )
        group.add_argument(
            "--postnet-layers", default=5, type=int, help="Number of postnet layers"
        )
        group.add_argument(
            "--postnet-chans", default=256, type=int, help="Number of postnet channels"
        )
        group.add_argument(
            "--postnet-filts", default=5, type=int, help="Filter size of postnet"
        )
        group.add_argument(
            "--use-scaled-pos-enc",
            default=False,
            type=strtobool,
            help="Use trainable scaled positional encoding "
            "instead of the fixed scale one.",
        )
        group.add_argument(
            "--use-batch-norm",
            default=True,
            type=strtobool,
            help="Whether to use batch normalization",
        )
        group.add_argument(
            "--encoder-normalize-before",
            default=False,
            type=strtobool,
            help="Whether to apply layer norm before encoder block",
        )
        group.add_argument(
            "--decoder-normalize-before",
            default=False,
            type=strtobool,
            help="Whether to apply layer norm before decoder block",
        )
        group.add_argument(
            "--encoder-concat-after",
            default=False,
            type=strtobool,
            help="Whether to concatenate attention layer's input and output in encoder",
        )
        group.add_argument(
            "--use-dur-predictor",
            default=False,
            type=strtobool,
            help="Whether to concatenate attention layer's input and output in encoder",
        )
        group.add_argument(
            "--duration-predictor-layers",
            default=2,
            type=int,
            help="Number of layers in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-chans",
            default=384,
            type=int,
            help="Number of channels in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-kernel-size",
            default=3,
            type=int,
            help="Kernel size in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for duration predictor",
        )
        group.add_argument(
            "--decoder-concat-after",
            default=False,
            type=strtobool,
            help="Whether to concatenate attention layer's input and output in decoder",
        )
        group.add_argument(
            "--reduction-factor", default=1, type=int, help="Reduction factor"
        )
        group.add_argument(
            "--spk-embed-dim",
            default=None,
            type=int,
            help="Number of speaker embedding dimensions",
        )
        group.add_argument(
            "--spk-embed-integration-type",
            type=str,
            default="add",
            choices=["add", "concat"],
            help="How to integrate speaker embedding",
        )
        # training related
        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="How to initialize transformer parameters",
        )
        group.add_argument(
            "--initial-encoder-alpha",
            type=float,
            default=1.0,
            help="Initial alpha value in encoder's ScaledPositionalEncoding",
        )
        group.add_argument(
            "--initial-decoder-alpha",
            type=float,
            default=1.0,
            help="Initial alpha value in decoder's ScaledPositionalEncoding",
        )
        group.add_argument(
            "--transformer-lr",
            default=1.0,
            type=float,
            help="Initial value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=4000,
            type=int,
            help="Optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-enc-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder except for attention",
        )
        group.add_argument(
            "--transformer-enc-positional-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder positional encoding",
        )
        group.add_argument(
            "--transformer-enc-attn-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder self-attention",
        )
        group.add_argument(
            "--transformer-dec-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer decoder "
            "except for attention and pos encoding",
        )
        group.add_argument(
            "--transformer-dec-positional-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer decoder positional encoding",
        )
        group.add_argument(
            "--transformer-dec-attn-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer decoder self-attention",
        )
        group.add_argument(
            "--transformer-enc-dec-attn-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for transformer encoder-decoder attention",
        )
        group.add_argument(
            "--eprenet-dropout-rate",
            default=0.5,
            type=float,
            help="Dropout rate in encoder prenet",
        )
        group.add_argument(
            "--use-rhythm-embed", default=True, type=strtobool, help="Number of postnet layers"
        )
        group.add_argument(
            "--dprenet-dropout-rate",
            default=0.5,
            type=float,
            help="Dropout rate in decoder prenet",
        )
        group.add_argument(
            "--postnet-dropout-rate",
            default=0.5,
            type=float,
            help="Dropout rate in postnet",
        )
        group.add_argument(
            "--pretrained-model", default=None, type=str, help="Pretrained model path"
        )
        # loss related
        group.add_argument(
            "--use-masking",
            default=True,
            type=strtobool,
            help="Whether to use masking in calculation of loss",
        )
        group.add_argument(
            "--use-weighted-masking",
            default=False,
            type=strtobool,
            help="Whether to use weighted masking in calculation of loss",
        )
        group.add_argument(
            "--loss-type",
            default="L2",
            choices=["L1", "L2", "L1+L2"],
            help="How to calc loss",
        )
        group.add_argument(
            "--bce-pos-weight",
            default=5.0,
            type=float,
            help="Positive sample weight in BCE calculation "
            "(only for use-masking=True)",
        )
        group.add_argument(
            "--use-guided-attn-loss",
            default=False,
            type=strtobool,
            help="Whether to use guided attention loss",
        )
        group.add_argument(
            "--guided-attn-loss-sigma",
            default=0.4,
            type=float,
            help="Sigma in guided attention loss",
        )
        group.add_argument(
            "--guided-attn-loss-lambda",
            default=1.0,
            type=float,
            help="Lambda in guided attention loss",
        )
        group.add_argument(
            "--num-heads-applied-guided-attn",
            default=2,
            type=int,
            help="Number of heads in each layer to be applied guided attention loss"
            "if set -1, all of the heads will be applied.",
        )
        group.add_argument(
            "--num-layers-applied-guided-attn",
            default=2,
            type=int,
            help="Number of layers to be applied guided attention loss"
            "if set -1, all of the layers will be applied.",
        )
        group.add_argument(
            "--modules-applied-guided-attn",
            type=str,
            nargs="+",
            default=["encoder-decoder"],
            help="Module name list to be applied guided attention loss",
        )
        group.add_argument(
            "--teacher-model",
            default=None,
            type=str,
            nargs="?",
            help="Teacher model file path",
        )
        return parser
       
    def _load_teacher_model(self, model_path):
        # get teacher model config
        idim, odim, arg = get_model_conf(model_path)

        # assert dimension is the same between teacher and studnet
        assert idim == self.idim
        assert odim == self.odim
        assert arg.reduction_factor == self.reduction_factor

        # load teacher model
        from espnet.utils.dynamic_import import dynamic_import
        

        model_class = dynamic_import(arg.model_module)
        model = model_class(idim, odim, arg)
       
        torch_load(model_path, model)
        model.eval()
        model = model.to('cuda')

        # freeze teacher model parameters
        for p in model.parameters():
            p.requires_grad = False

        return model

    @property
    def attention_plot_class(self):
        """Return plot class for attention weight plot."""
        return TTSPlot

    def __init__(self, idim, odim, args=None):
        """Initialize TTS-Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - embed_dim (int): Dimension of character embedding.
                - eprenet_conv_layers (int):
                    Number of encoder prenet convolution layers.
                - eprenet_conv_chans (int):
                    Number of encoder prenet convolution channels.
                - eprenet_conv_filts (int): Filter size of encoder prenet convolution.
                - dprenet_layers (int): Number of decoder prenet layers.
                - dprenet_units (int): Number of decoder prenet hidden units.
                - elayers (int): Number of encoder layers.
                - eunits (int): Number of encoder hidden units.
                - adim (int): Number of attention transformation dimensions.
                - aheads (int): Number of heads for multi head attention.
                - dlayers (int): Number of decoder layers.
                - dunits (int): Number of decoder hidden units.
                - postnet_layers (int): Number of postnet layers.
                - postnet_chans (int): Number of postnet channels.
                - postnet_filts (int): Filter size of postnet.
                - use_scaled_pos_enc (bool):
                    Whether to use trainable scaled positional encoding.
                - use_batch_norm (bool):
                    Whether to use batch normalization in encoder prenet.
                - encoder_normalize_before (bool):
                    Whether to perform layer normalization before encoder block.
                - decoder_normalize_before (bool):
                    Whether to perform layer normalization before decoder block.
                - encoder_concat_after (bool): Whether to concatenate attention
                    layer's input and output in encoder.
                - decoder_concat_after (bool): Whether to concatenate attention
                    layer's input and output in decoder.
                - reduction_factor (int): Reduction factor.
                - spk_embed_dim (int): Number of speaker embedding dimenstions.
                - spk_embed_integration_type: How to integrate speaker embedding.
                - transformer_init (float): How to initialize transformer parameters.
                - transformer_lr (float): Initial value of learning rate.
                - transformer_warmup_steps (int): Optimizer warmup steps.
                - transformer_enc_dropout_rate (float):
                    Dropout rate in encoder except attention & positional encoding.
                - transformer_enc_positional_dropout_rate (float):
                    Dropout rate after encoder positional encoding.
                - transformer_enc_attn_dropout_rate (float):
                    Dropout rate in encoder self-attention module.
                - transformer_dec_dropout_rate (float):
                    Dropout rate in decoder except attention & positional encoding.
                - transformer_dec_positional_dropout_rate (float):
                    Dropout rate after decoder positional encoding.
                - transformer_dec_attn_dropout_rate (float):
                    Dropout rate in deocoder self-attention module.
                - transformer_enc_dec_attn_dropout_rate (float):
                    Dropout rate in encoder-deocoder attention module.
                - eprenet_dropout_rate (float): Dropout rate in encoder prenet.
                - dprenet_dropout_rate (float): Dropout rate in decoder prenet.
                - postnet_dropout_rate (float): Dropout rate in postnet.
                - use_masking (bool):
                    Whether to apply masking for padded part in loss calculation.
                - use_weighted_masking (bool):
                    Whether to apply weighted masking in loss calculation.
                - bce_pos_weight (float): Positive sample weight in bce calculation
                    (only for use_masking=true).
                - loss_type (str): How to calculate loss.
                - use_guided_attn_loss (bool): Whether to use guided attention loss.
                - num_heads_applied_guided_attn (int):
                    Number of heads in each layer to apply guided attention loss.
                - num_layers_applied_guided_attn (int):
                    Number of layers to apply guided attention loss.
                - modules_applied_guided_attn (list):
                    List of module names to apply guided attention loss.
                - guided-attn-loss-sigma (float) Sigma in guided attention loss.
                - guided-attn-loss-lambda (float): Lambda in guided attention loss.

        """
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)
        
        

        # store hyperparameters
        self.stage=args.train_stage
        self.feat_pad_value=args.feat_pad_value
        self.use_rhythm_embed=args.use_rhythm_embed
        self.in_global =args.inference_global
        self.dur_classfication=args.dur_classfication
        self.dur_self_attn=args.dur_self_attn
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = args.spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = args.spk_embed_integration_type
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.reduction_factor = args.reduction_factor
        self.loss_type = args.loss_type
        self.dur_frame_buffer=args.dur_frame_buffer
        self.use_guided_attn_loss = args.use_guided_attn_loss
        if self.use_guided_attn_loss:
            if args.num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = args.elayers
            else:
                self.num_layers_applied_guided_attn = (
                    args.num_layers_applied_guided_attn
                )
            if args.num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = args.aheads
            else:
                self.num_heads_applied_guided_attn = args.num_heads_applied_guided_attn
            self.modules_applied_guided_attn = args.modules_applied_guided_attn

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define transformer encoder
        if args.eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    rhythm_idim=args.rhythm_num,
                    embed_rhythm_dim=args.embed_rhythm_dim,
                    concat_dim=args.concat_dim,
                    embed_dim=args.embed_dim,
                    elayers=0,
                    econv_layers=args.eprenet_conv_layers,
                    econv_chans=args.eprenet_conv_chans,
                    econv_filts=args.eprenet_conv_filts,
                    use_batch_norm=args.use_batch_norm,
                    dropout_rate=args.eprenet_dropout_rate,
                    padding_idx=padding_idx,
                ),
                torch.nn.Linear(args.eprenet_conv_chans, args.adim),
            )

        else:

            encoder_input_layer=Rhythm_Embedding(indim=idim,
            outdim=args.adim,
            rhythm_num=args.rhythm_num,
            embed_rhythm_dim=args.embed_rhythm_dim,
            concat_dim=args.concat_dim,
            use_rhythm_embed=self.use_rhythm_embed)
        self.teacher_model=args.teacher_model

        self.encoder = Encoder_v2(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after,
            positionwise_layer_type=args.positionwise_layer_type,
            positionwise_conv_kernel_size=args.positionwise_conv_kernel_size,
        )
        self.length_regulator = LengthRegulator_gs()
        if self.dur_self_attn:
            self.dur_fft_block=fft_block(384,384,0.1,4)

        if args.use_dur_predictor:  
        
            self.duration_predictor = Duration_variance_Predictor(
            idim=args.adim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
            )
            
        
        
        # define projection layer
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, args.adim)
            else:
                self.projection = torch.nn.Linear(
                    args.adim + self.spk_embed_dim, args.adim
                )
        

        # define transformer decoder
        if args.dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=odim,
                    n_layers=args.dprenet_layers,
                    n_units=args.dprenet_units,
                    dropout_rate=args.dprenet_dropout_rate,
                ),
                torch.nn.Linear(args.dprenet_units, args.adim),
            )
        else:
            decoder_input_layer = "linear"
        self.decoder=Phone_Autoaggressive_Decoder(
            odim=-1,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=True,

        )
        
        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)

        # define postnet
        self.postnet = (
            None
            if args.postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=args.postnet_layers,
                n_chans=args.postnet_chans,
                n_filts=args.postnet_filts,
                use_batch_norm=args.use_batch_norm,
                dropout_rate=args.postnet_dropout_rate,
            )
        )

        # define loss function
        self.use_dur_predictor=args.use_dur_predictor
 
        if args.use_dur_predictor:
            if self.dur_classfication:
                 self.criterion=FeedForwardTransformerLoss_class(
                    use_masking=args.use_masking,
                    use_weighted_masking=args.use_weighted_masking,
                    stage=self.stage
                    
                )           
            else:
                self.criterion=FeedForwardTransformerLoss(
                    use_masking=args.use_masking,
                    use_weighted_masking=args.use_weighted_masking,
                    stage=self.stage
                    
                )
        
        else:
            self.criterion = TransformerLoss(
                use_masking=args.use_masking,
                use_weighted_masking=args.use_weighted_masking,
                bce_pos_weight=args.bce_pos_weight,
            )
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=args.guided_attn_loss_sigma, alpha=args.guided_attn_loss_lambda,
             
            )
        print(get_parameter_number(self.encoder))
        print(get_parameter_number(self.decoder))
        print(get_parameter_number(self.postnet))
        print(get_parameter_number(self.feat_out))

        # initialize parameters
        # self._reset_parameters(
            # init_type=args.transformer_init,
            # init_enc_alpha=args.initial_encoder_alpha,
            # init_dec_alpha=args.initial_decoder_alpha,
        # )

        # load pretrained model
        if args.pretrained_model is not None:
            self.load_pretrained_model(args.pretrained_model)

    def init_teacher_model(self):
           # define teacher model
        if self.teacher_model is not None:
            self.teacher = self._load_teacher_model(self.teacher_model)
            
        else:
            self.teacher = None

        # define duration calculator
        if self.teacher is not None:
            self.duration_calculator = DurationCalculator(self.teacher)
            
        else:
            self.duration_calculator = None
    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
            
    def _get_teacher_model_dur(self,xs, ys):
        spembs=None
        y=ys
        y=self._add_first_frame_and_remove_last_frame(y)
        ilens=torch.LongTensor([xs.shape[2]]).to(xs.device)
        olens=torch.LongTensor([y.shape[1]]).to(xs.device)
        with torch.no_grad():
            
            ds = self.duration_calculator(
                        xs, ilens, y, olens, spembs
                    )  #
        return ds

    def _add_first_frame_and_remove_last_frame(self, ys):
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

    def forward(self, xs, ilens, ys, labels, olens, max_ys,spembs=None,pad_masks=None,phone_masks=None ,*args, **kwargs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            Tensor: Loss value.

        """
        # remove unnecessary padded part (for multi-gpus)
        
        pad_mask=pad_masks
        phone_mask=phone_masks
   
        max_ys=max_ys.item()+self.dur_frame_buffer
        
 
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
  

        # forward encoder
        xs_input=xs[:,:2,:].view(xs.shape[0],2,-1)
        x_masks = self._source_mask(ilens)
        src_pos=self.get_seq_mask(ilens).cuda()
        mel_pos=self.get_seq_mask(olens)
        mel_pos=torch.cat([mel_pos,torch.zeros_like(mel_pos[:,:self.dur_frame_buffer])],dim=1).cuda()
        HS, h_masks = self.encoder(xs_input,src_pos, x_masks)
        

        dur=xs[:,2,:].view(xs.shape[0],xs.shape[-1])
        d_masks = make_pad_mask(ilens).to(xs.device)
        d_outs , variance = self.duration_predictor(HS, d_masks)
        hs=self.length_regulator (xs=HS,dur=dur,variance=variance,ilen=ilens,olen=olens,i_max=HS.shape[1],o_max=max_ys)

        query_mask=self._source_mask(olens).squeeze(1)
        query_mask=torch.cat([query_mask,torch.zeros_like(query_mask[:,:self.dur_frame_buffer])],dim=-1)
        if self.dur_self_attn:
            a_mask=query_mask.unsqueeze(1).repeat(1,query_mask.shape[1], 1).eq(0).repeat(4, 1, 1)
            hs=self.dur_fft_block( query=hs , attn_mask=a_mask, query_mask=query_mask)[0]

 

        # add first zero frame and remove last frame for auto-regressive
        ys_in = torch.cat([torch.ones_like(ys[:,:self.dur_frame_buffer,:])*self.feat_pad_value,ys],dim=1)
        ys=torch.cat([ys,torch.ones_like(ys[:,:self.dur_frame_buffer,:])*self.feat_pad_value],dim=1)

        # forward decoder
        

        zs, _ ,__=self.decoder(query=ys_in,pos=mel_pos,memory=hs,pad_mask=pad_mask,dur_mask=phone_mask,query_mask=query_mask)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)
            
  
        # caluculate loss values
        if self.use_dur_predictor:
            d_masks = make_pad_mask(ilens).to(xs.device)
            d_outs , variance = self.duration_predictor(HS, d_masks)

            l1_loss, duration_loss = self.criterion(
                after_outs, before_outs, d_outs, ys, dur, ilens-1, olens
            )
            loss = l1_loss + duration_loss*0.1
            report_keys = [
            {"l1_loss": l1_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"loss": loss.item()},
            ]

            
        else:
            l1_loss, l2_loss = self.criterion(
                after_outs, before_outs, ys, olens
            )
            if self.loss_type == "L1":
                loss = l1_loss
            elif self.loss_type == "L2":
                loss = l2_loss
            elif self.loss_type == "L1+L2":
                loss = l1_loss + l2_loss
            else:
                raise ValueError("unknown --loss-type " + self.loss_type)
            report_keys = [
                {"l1_loss": l1_loss.item()},
                {"l2_loss": l2_loss.item()},
                {"loss": loss.item()},
            ]

        # calculate guided attention loss

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, spemb=None,y=None ,*args, **kwargs):
        """Generate the sequence of features given the sequences of characters.

        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace):
                - threshold (float): Threshold in inference.
                - minlenratio (float): Minimum length ratio in inference.
                - maxlenratio (float): Maximum length ratio in inference.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Encoder-decoder (source) attention weights (#layers, #heads, L, T).

        """

        device=x.device

        # forward encoder
        if y is  None : 
            xs = x.unsqueeze(0)
            xs_input=xs[:,:2,:].view(xs.shape[0],2,-1)
            src_pos=torch.arange(1,xs.shape[-1]+1,device=device)
            HS, _ = self.encoder(xs_input,src_pos, None)
          
            if self.use_dur_predictor:
                D, variance=self.duration_predictor.inference(HS,None)
                D=D.float() 
            else:
                D=xs[:,2,:].view(xs.shape[0],xs.shape[-1]).float()  
            D=xs[:,2,:].view(xs.shape[0],xs.shape[-1]).float()
            dur=torch.clamp(D,0,self.dur_frame_buffer).long()
            
                    
            expand_hs=self.length_regulator (xs=HS,dur=dur,variance=variance,ilen=None,olen=None,i_max=None,o_max=None).squeeze(0)
            
            
           
            
            if self.dur_self_attn:
                qm=torch.sum(dur,dim=1)
                qm=torch.ones([1,qm]).to(device)
                a_mask=qm.unsqueeze(1).repeat(1,qm.shape[1], 1).eq(0).repeat(4, 1, 1)
                expand_hs=self.dur_fft_block(query=expand_hs.unsqueeze(0) , attn_mask=a_mask, query_mask=qm)[0].squeeze(0)
            dur_numpy=dur.detach().to('cpu').numpy()
            dur_cumsum=dur.cumsum(dim=-1).view(-1)
            ys_in=torch.ones([xs.shape[0],self.dur_frame_buffer,self.odim],device=device)*self.feat_pad_value
            steps=dur.shape[1]
            #caculate mask

            pad_mask_all=torch.from_numpy(get_pad_mask(dur_numpy,self.dur_frame_buffer,False)).to(device).squeeze(0)
            phone_mask_all=torch.from_numpy(get_phone_mask(dur_numpy)).to(device).squeeze(0)   
            
            
            
            if self.in_global:
                decoder_cache=None
            else:    
                decoder_cache=[None]*(self.decoder.decoders.decoder_num+1)
                
            for i in range(steps-1):
                
                mel_max_len=torch.sum(dur[0,:i])+self.dur_frame_buffer
                   
                hs=expand_hs[:dur_cumsum[i],:]
                hs=F.pad(hs,[0,0,0,mel_max_len-dur_cumsum[i]]).unsqueeze(0)
                query_mask= torch.cat([torch.arange(1, dur_cumsum[i] + 1,device=device).unsqueeze(0),
                                  torch.zeros([1,mel_max_len-dur_cumsum[i]],device=device).long()],
                                  dim=-1)
                                                                   
                phone_mask=phone_mask_all[:dur_cumsum[i],:dur_cumsum[i]]
                pad_mask=pad_mask_all[:dur_cumsum[i],:mel_max_len]
                phone_mask=F.pad(phone_mask,[0,mel_max_len-dur_cumsum[i],0,mel_max_len-dur_cumsum[i]]).unsqueeze(0)
                pad_mask=F.pad(pad_mask,[0,0,0,mel_max_len-dur_cumsum[i]]).unsqueeze(0)
                
                if not self.in_global:
                    pad_mask=pad_mask[:,-50:]
                    phone_mask=phone_mask[:,-50:]
                    query_mask=query_mask[:,-50:]
                 
                    
                query_mask=query_mask.ne(0)
                mel_pos = torch.arange(1,1+mel_max_len).to(device)
                mel_pos=torch.clamp(mel_pos,0,1023).long()
                zs, _,new_cache =self.decoder(query=ys_in, 
                                    pos=mel_pos,
                                    memory=hs,
                                    pad_mask=pad_mask,
                                    dur_mask=phone_mask,
                                    query_mask=query_mask,
                                    decoder_cache=decoder_cache)
                if self.in_global :        
                    before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
                    
                    if self.postnet is None:
                        after_outs = before_outs
                    else:
                        after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)                    
                    ys=after_outs                
                    if i <steps-2:
                        ys_in=torch.cat([torch.ones([ys.shape[0],self.dur_frame_buffer,self.odim],device=device)*self.feat_pad_value,
                                        ys[:,:dur_cumsum[i],:]],dim=1)                         
                    else:
                        ys=ys[:,:dur_cumsum[i],:]
                        break;
                    
                else :
                     before_outs = self.feat_out(zs[:,-self.dur_frame_buffer:-self.dur_frame_buffer+dur[0,i],:]).view(zs.size(0), -1, self.odim)
                     if self.postnet is None:
                        after_outs = before_outs
                     else:
                        after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)     
                     if decoder_cache[-1] is None :
                        decoder_cache[-1]=after_outs
                     else :
                        decoder_cache[-1]=torch.cat([decoder_cache[-1],after_outs],dim=1)
                     
                     for j ,cache in enumerate(new_cache):
                        decoder_cache[j]=cache[:,:dur_cumsum[i],:]
                     if i<steps-2:
                        ys_in=torch.cat([torch.ones([ys_in.shape[0],self.dur_frame_buffer,self.odim],device=device)*self.feat_pad_value,
                                        decoder_cache[-1]],dim=1)                         
                     else:
                        ys=decoder_cache[-1]
                        break
                                             
            
        else:
           
            xs = x.unsqueeze(0)
            y  = y.unsqueeze(0)        
            xs_input=xs[:,:2,:].view(xs.shape[0],2,-1)
            src_pos=torch.arange(1,xs.shape[-1]+1,device=device).view(1,-1)
            hs, _ = self.encoder(xs_input,src_pos, None)
            dur=xs[:,2,:].view(xs.shape[0],xs.shape[-1])
            ys_in = torch.cat([torch.ones_like(y[:,:self.dur_frame_buffer,:])*self.feat_pad_value,y],dim=1)
            hs=self.length_regulator(hs,dur,y.shape[1]+self.dur_frame_buffer)
            olens=torch.from_numpy(np.asarray(y.shape[1])).view(1)
            query_mask=self._source_mask(olens).squeeze(1)
            query_mask=torch.cat([query_mask,torch.zeros_like(query_mask[:,:self.dur_frame_buffer])],dim=-1).to(device)
            pad_mask=torch_get_pad_mask(dur,self.dur_frame_buffer)
            phone_mask=torch_get_phone_mask(dur[:,:-1],y.shape[1]+self.dur_frame_buffer)
            mel_pos=self.get_seq_mask(olens)
            mel_pos=torch.cat([mel_pos,torch.zeros_like(mel_pos[:,:self.dur_frame_buffer])],dim=1).to(device)
            zs, _ ,__=self.decoder(ys_in,pos=mel_pos, memory=hs,pad_mask=pad_mask,dur_mask=phone_mask,query_mask=query_mask)
            before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
            if self.postnet is None:
                after_outs = before_outs
            else:
                after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)
                
            ys= after_outs[:,:torch.sum(dur[0,:-1],dim=-1),:]
           
            
        
        outs=ys


        
    
     
        
        return {'mel_feats':outs.squeeze(0),
                'dur':dur.squeeze().detach().cpu().numpy(),
                'input_id':x[0,:].squeeze().detach().to('cpu').numpy()}


    def calculate_all_attentions(
        self,
        xs,
        ilens,
        ys,
        olens,
        max_ys,
        spembs=None,
        skip_output=False,
        keep_tensor=False,
        *args,
        **kwargs
    ):
        """Calculate all of the attention weights.

        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).
            skip_output (bool, optional): Whether to skip calculate the final output.
            keep_tensor (bool, optional): Whether to keep original tensor.

        Returns:
            dict: Dict of attention weights and outputs.

        """
        with torch.no_grad():
            # forward encoder
            max_ys=max_ys.item()+self.dur_frame_buffer
            xs_input=xs[:,:2,:].view(xs.shape[0],2,-1)
            x_masks = self._source_mask(ilens)
                    
            src_pos=self.get_seq_mask(ilens).cuda()
            mel_pos=self.get_seq_mask(olens)
            mel_pos=torch.cat([mel_pos,torch.zeros_like(mel_pos[:,:self.dur_frame_buffer])],dim=1).cuda()
            hs, h_masks = self.encoder(xs_input,src_pos, x_masks)
            dur=xs[:,2,:].view(xs.shape[0],xs.shape[-1])
            d_masks = make_pad_mask(ilens).to(xs.device)      
            d_outs , variance = self.duration_predictor(hs, d_masks)
            hs=self.length_regulator(xs=hs,dur=dur,variance=variance,ilen=ilens,olen=olens,i_max=hs.shape[1],o_max=max_ys)

            # integrate speaker embedding
            if self.spk_embed_dim is not None:
                hs = self._integrate_with_spk_embed(hs, spembs)

            # thin out frames for reduction factor
            # (B, Lmax, odim) ->  (B, Lmax//r, odim)
            query_mask=self._source_mask(olens).squeeze(1)
            query_mask=torch.cat([query_mask,torch.zeros_like(query_mask[:,:self.dur_frame_buffer])],dim=-1)
            if self.dur_self_attn:
                a_mask=query_mask.unsqueeze(1).repeat(1,query_mask.shape[1], 1).eq(0).repeat(4, 1, 1)
                hs=self.dur_fft_block( query=hs , attn_mask=a_mask, query_mask=query_mask)[0]


            pad_mask=torch_get_pad_mask(dur,self.dur_frame_buffer)
            phone_mask=torch_get_phone_mask(dur[:,:-1],max_ys)
            
            olens_in = olens

            # add first zero frame and remove last frame for auto-regressive
            ys_in = torch.cat([torch.ones_like(ys[:,:self.dur_frame_buffer,:])*self.feat_pad_value,ys],dim=1)

            # forward decoder
            
            zs, _ ,_=self.decoder(ys_in,pos=mel_pos,memory=hs,pad_mask=pad_mask,dur_mask=phone_mask,query_mask=query_mask)

            # calculate final outputs
            if not skip_output:
                before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
                if self.postnet is None:
                    after_outs = before_outs
                else:
                    after_outs = before_outs + self.postnet(
                        before_outs.transpose(1, 2)
                    ).transpose(1, 2)

        # modifiy mod part of output lengths due to reduction factor > 1
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])

        # store into dict
        att_ws_dict = dict()
        if keep_tensor:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    att_ws_dict[name] = m.attn
            if not skip_output:
                att_ws_dict["before_postnet_fbank"] = before_outs
                att_ws_dict["after_postnet_fbank"] = after_outs
        else:
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention):
                    attn = m.attn.cpu().numpy()
                    if "encoder" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                    elif "decoder" in name:
                        if "src" in name:
                            attn = [
                                a[:, :ol, :il]
                                for a, il, ol in zip(
                                    attn, ilens.tolist(), olens_in.tolist()
                                )
                            ]
                        elif "self" in name:
                            attn = [
                                a[:, :l, :l] for a, l in zip(attn, olens_in.tolist())
                            ]
                        else:
                            logging.warning("unknown attention module: " + name)
                    else:
                        logging.warning("unknown attention module: " + name)
                    att_ws_dict[name] = attn
            if not skip_output:
                before_outs = before_outs.cpu().numpy()
                after_outs = after_outs.cpu().numpy()
                att_ws_dict["before_postnet_fbank"] = [
                    m[:l].T for m, l in zip(before_outs, olens.tolist())
                ]
                att_ws_dict["after_postnet_fbank"] = [
                    m[:l].T for m, l in zip(after_outs, olens.tolist())
                ]

        return att_ws_dict

    def _integrate_with_spk_embed(self, hs, spembs):
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
        
    def get_seq_mask(self,length):

        #length :shape of [B,]
        mask=[]
        max_t=torch.max(length)
        for i in length:
            mask.append(torch.nn.functional.pad(torch.arange(1,i+1),[0,max_t-i]))
        mask=torch.stack(mask,dim=0)
        return mask
        



    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _target_mask(self, olens):
        """Make masks for masked self-attention.

        Args:
            olens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training.

        keys should match what `chainer.reporter` reports.
        If you add the key `loss`, the reporter will report `main/loss`
        and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss`
        and `validation/main/loss` values.

        Returns:
            list: List of strings which are base keys to plot during training.

        """
        plot_keys = ["loss", "l1_loss", "l2_loss", "bce_loss",'duration_loss']
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]
        if self.use_guided_attn_loss:
            if "encoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_attn_loss"]
            if "decoder" in self.modules_applied_guided_attn:
                plot_keys += ["dec_attn_loss"]
            if "encoder-decoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_dec_attn_loss"]

        return plot_keys
