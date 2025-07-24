import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiEncoder, MimiTransformerModel, MimiConv1d, MimiConvTranspose1d, MimiDecoder
from nemo.core.classes.module import NeuralModule
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

from nemo.collections.common.parts.utils import ClampActivation
from nemo.collections.tts.modules.audio_codec_modules import CodecActivation, CausalConv1dNorm




from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int = 1152,
        layer_scale_init_value: float = 1.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.dwconv = CausalConv1D(dim, dim, kernel_size=7, padding=None, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

class TemporalSmoothingHead(nn.Module):
    def __init__(self, in_channels=882, kernel_size=5, out_kernel_size=3, pad_mode="zeros", output_activation="clamp", activation="half_snake"):
        super().__init__()
        
        self.pre_conv = CausalConv1dNorm(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, pad_mode=pad_mode)
        self.pre_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = CausalConv1dNorm(in_channels=in_channels, out_channels=in_channels, kernel_size=out_kernel_size, pad_mode=pad_mode)
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()

    def forward(self, x, x_len):
        out = x.transpose(1, 2)
        out = self.pre_conv(inputs=out, input_len=x_len)
        out = self.pre_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=x_len)
        out = self.out_activation(out)
        return out.transpose(1, 2)

from contextlib import contextmanager
@contextmanager
def default_precision(dtype=torch.float32):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)


class ReshapeTransformerEncoder(NeuralModule):
    """
    Transformer Audio encoder.

    Args:
        output_dim: Dimension of encoder output.
    """

    def __init__(
        self,
        samples_per_frame: int,
        audio_proj_size: int = 1024, 
        output_dim: int = 32,
        n_layers: int = 8,
        d_model: int = 1024,
        d_ffn: int = 4096,
        is_causal: bool = True,
        sliding_window_size: int = 12,
        max_position_embeddings: int = 8000,
        rope_theta: float = 10000.0,
        attn_implementation: str = "eager",
    ):
        super().__init__()

        self.is_causal = is_causal
        self.samples_per_frame = samples_per_frame
        self.audio_proj_size = audio_proj_size
        self.output_dim = output_dim

        self.config = MimiConfig()
        self.config._attn_implementation = attn_implementation
        self.config.max_position_embeddings = max_position_embeddings
        self.config.rope_theta = rope_theta

        self.config.use_causal_conv = is_causal
        self.config.num_hidden_layers = n_layers
        self.config.intermediate_size = d_ffn
        self.config.hidden_size = d_model
        self.config.sliding_window = sliding_window_size
        self.layers = MimiTransformerModel(self.config)

        self.inp_projection_no_bias = nn.Linear(samples_per_frame, audio_proj_size, bias=False)
        self.inp_projection = nn.Linear(audio_proj_size, d_model)
        self.out_projection = nn.Linear(d_model, output_dim)

    def forward(self, audio, audio_len):
        encoded_len = audio_len
        B, T = audio.size()
        audio = audio.reshape(B, -1, self.samples_per_frame) # B, T, F, where 7 is the number of samples per frame that controls the frame rate
        with default_precision(torch.float32):
            encoded_len = (audio_len / self.samples_per_frame).long()

        if self.is_causal:
            mask = get_mask_from_lengths(encoded_len)
        else:
            # mask none does not apply causal mask
            mask = None

        out = self.inp_projection_no_bias(audio)
        out = self.inp_projection(out)

        out = self.layers(out, attention_mask=mask)[0]
        # out projection
        encoded = self.out_projection(out).transpose(1, 2)
        return encoded, encoded_len


class ReshapeTransformerDecoder(NeuralModule):
    """
    Transformer Audio Decoder.

    Args:
        input_dim: Dimension of encoder output.
    """

    def __init__(
        self,
        samples_per_frame: int,
        audio_proj_size: int = 1024, 
        input_dim: int = 32,
        n_layers: int = 8,
        d_model: int = 1024,
        d_ffn: int = 4096,
        is_causal: bool = True,
        sliding_window_size: int = 12,
        max_position_embeddings: int = 8000,
        rope_theta: float = 10000.0,
        attn_implementation: str = "eager",
        use_conv_pos: bool = False,
        num_pos_conv_blocks: int = 1,
        use_temporal_smoth_head: bool = False,
    ):
        super().__init__()

        self.samples_per_frame = samples_per_frame
        self.audio_proj_size = audio_proj_size
        self.is_causal = is_causal
        self.use_conv_pos = use_conv_pos
        self.use_temporal_smoth_head = use_temporal_smoth_head

        self.config = MimiConfig()
        self.config._attn_implementation = attn_implementation
        self.config.max_position_embeddings = max_position_embeddings
        self.config.rope_theta = rope_theta

        self.config.use_causal_conv = is_causal
        self.config.num_hidden_layers = n_layers
        self.config.intermediate_size = d_ffn
        self.config.hidden_size = d_model
        self.config.sliding_window = sliding_window_size
        self.layers = MimiTransformerModel(self.config)

        self.inp_projection = nn.Linear(input_dim, d_model)
        self.out_projection = nn.Linear(d_model, audio_proj_size)
        self.out_projection_no_bias = nn.Linear(audio_proj_size, samples_per_frame, bias=False)

        # add ConvNeXt based conv pos
        if self.use_conv_pos:
            self.conv_pos = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=d_model
                )
                for _ in range(num_pos_conv_blocks)
            ]
        )

        if self.use_temporal_smoth_head:
            self.temporal_smoth_head = TemporalSmoothingHead(d_model)

    def forward(self, inputs, input_len):
        if self.is_causal:
            mask = get_mask_from_lengths(input_len)
        else:
            # mask none does not apply causal mask
            mask = None

        encoded_len = input_len
        out = self.inp_projection(inputs.transpose(1, 2))
        out = self.layers(out, attention_mask=mask)[0]

        if self.use_conv_pos:
            out = out.transpose(1, 2)
            for conv_block in self.conv_pos:
                out = conv_block(out)
            out = out.transpose(1, 2)

        if self.use_temporal_smoth_head:
            out = self.temporal_smoth_head(out, input_len)

        out = self.out_projection(out)
        audio = self.out_projection_no_bias(out)

        # resample audio to size
        audio = audio.reshape(inputs.size(0), -1)
        audio_len = (input_len*self.samples_per_frame).int()
        return audio, audio_len


class MimiAudioEncoder(NeuralModule):
    def __init__(self, out_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512, sliding_window=250, num_transformer_layers=8):
        super().__init__()
        self.is_causal = is_causal

        # get Mimi default config
        self.config = MimiConfig()
        self.config._attn_implementation = "eager"

        # redefine configs based on nemo configs
        self.config.frame_rate = frame_rate
        self.config.sampling_rate = sampling_rate
        self.config.upsampling_ratios = upsampling_ratios
        self.config.use_causal_conv = is_causal
        self.config.hidden_size = hidden_size
        self.config.sliding_window = sliding_window
        self.config.num_hidden_layers = num_transformer_layers

        # define upsampling rate
        self.downsampling_rate = self.config.sampling_rate / self.config.frame_rate

        self.encoder = MimiEncoder(self.config)
        self.encoder_transformer = MimiTransformerModel(self.config)

        # extra downsample requeried because MiMiEncoder works in a different frame rate
        self.use_extra_downsample = self.encodec_frame_rate != self.config.frame_rate
        if self.use_extra_downsample:
            self.downsample = MimiConv1d(
                self.config,
                self.config.hidden_size,
                self.config.hidden_size,
                kernel_size=2 * int(self.encodec_frame_rate / self.config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
            )

        self.out_projection = MimiConv1d(
            self.config,
            self.config.hidden_size,
            out_size,
            kernel_size=1,
            stride=1,
            bias=False,
            pad_mode="replicate",
        )

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.config.upsampling_ratios)
        return math.ceil(self.config.sampling_rate / hop_length)

    def forward(self, audio, audio_len):
        if self.is_causal:
            mask = get_mask_from_lengths(audio_len)
        else:
            # mask none does not apply causal mask
            mask = None

        audio = audio.unsqueeze(1)
        embeddings = self.encoder(audio)
        embeddings = self.encoder_transformer(
            embeddings.transpose(1, 2), attention_mask=mask
        )[0].transpose(1, 2)

        if self.use_extra_downsample:
            embeddings = self.downsample(embeddings)

        embeddings = self.out_projection(embeddings)

        # compute output_len based on downsampling rate
        output_len = (audio_len / self.downsampling_rate).long()
        return embeddings, output_len


class MimiAudioDecoder(NeuralModule):
    def __init__(self, input_size=32, sampling_rate=24000, upsampling_ratios=[8, 6, 5, 4], frame_rate=12.5, is_causal=True, hidden_size=512, sliding_window=250, num_transformer_layers=8):
        super().__init__()
        self.is_causal = is_causal

        # get Mimi default config
        self.config = MimiConfig()
        self.config._attn_implementation = "eager"

        # redefine configs based on nemo configs
        self.config.frame_rate = frame_rate
        self.config.sampling_rate = sampling_rate
        self.config.upsampling_ratios = upsampling_ratios
        self.config.use_causal_conv = is_causal
        self.config.hidden_size = hidden_size
        self.config.sliding_window = sliding_window
        self.config.num_hidden_layers = num_transformer_layers

        # define upsampling rate
        self.upsampling_rate = self.config.sampling_rate / self.config.frame_rate

        self.decoder_transformer = MimiTransformerModel(self.config)
        self.decoder = MimiDecoder(self.config)

        # extra upsampling requeried because MiMiEncoder works in a different frame rate
        self.use_extra_upsample = self.encodec_frame_rate != self.config.frame_rate
        if self.use_extra_upsample:
            self.upsample = MimiConvTranspose1d(
                self.config,
                self.config.hidden_size,
                self.config.hidden_size,
                kernel_size=2 * int(self.encodec_frame_rate / self.config.frame_rate),
                stride=2,
                bias=False,
                groups=self.config.upsample_groups,
            )

        self.in_projection = MimiConv1d(
            self.config,
            input_size,
            self.config.hidden_size,
            kernel_size=1,
            stride=1,
            bias=False,
            pad_mode="replicate",
        )

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.config.upsampling_ratios)
        return math.ceil(self.config.sampling_rate / hop_length)

    def forward(self, inputs, input_len, past_key_values=None, return_dict=None, return_past_key_values=False):
        if self.is_causal:
            mask = get_mask_from_lengths(input_len)
        else:
            # mask none does not apply causal mask
            mask = None

        embeddings = self.in_projection(inputs)
        if self.use_extra_upsample:
            embeddings = self.upsample(embeddings)

        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), attention_mask=mask, past_key_values=past_key_values, return_dict=return_dict
        )

        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(embeddings).squeeze(1)
        # compute output len based on the upsampling rate
        output_len = (input_len * self.upsampling_rate).long()
        if return_past_key_values:
            if return_dict:
                past_key_values = decoder_outputs.get("past_key_values")
            elif len(decoder_outputs) > 1:
                past_key_values = decoder_outputs[1]
            return outputs, past_key_values
        return outputs, output_len

# Debug
# mimiencoder = MimiAudioEncoder()
# audio = torch.ones([2, 48000])
# audio_len = torch.zeros(audio.size(0))
# audio_len = audio_len + audio.size(1)
# unquantized_latent, unquantized_latent_len = mimiencoder(audio, audio_len)
# print("unquantized_latent:", unquantized_latent.shape, unquantized_latent_len)
# mimidecoder = MimiAudioDecoder()
# audio_out = mimidecoder(unquantized_latent, unquantized_latent_len)
# print("Audio output", audio_out[0].shape, audio_out[1])
# convert checkpoint
"""
import torch
from transformers import MimiModel
model = MimiModel.from_pretrained("kyutai/mimi")
state_dict = model.state_dict()
for key in list(state_dict.keys()):
    if "encoder." in key or "encoder_transformer." in key or "downsample." in key:
        state_dict["audio_encoder."+key] = state_dict[key]
        del state_dict[key]
    elif "decoder." in key or "decoder_transformer." in key or "upsample." in key:
        state_dict["audio_decoder."+key] = state_dict[key]
        del state_dict[key]
    elif "quantizer." in key:
        del state_dict[key]
    else:
        print("Key not converted!", key)

print(state_dict.keys())
state_dict_new = {'state_dict':state_dict} 
torch.save(state_dict, "/home/ecasanova/Projects/Checkpoints/MimiCodec/mimi_converted_to_nemo.ckpt")
"""