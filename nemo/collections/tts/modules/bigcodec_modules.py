import copy
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d
from einops import rearrange


def stft(x, fft_size, hop_size, win_length, window, use_complex=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device),
                        return_complex=True)

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(torch.clamp(
            x_stft.real ** 2 + x_stft.imag ** 2, min=1e-7, max=1e3)).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3) # [B, 2, T, F]
        return res

class SpecDiscriminator(nn.Module):
    def __init__(self,
                 stft_params=None,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=(7, 3),
                 channels=32,
                 max_downsample_channels=512,
                 downsample_scales=(2, 2, 2),
                 use_weight_norm=True,
                 spec_scale_pow=0.0,
                 ):
        super().__init__()
        self.use_weight_norm = use_weight_norm
        self.spec_scale_pow = spec_scale_pow

        if stft_params is None:
            stft_params = {
                'fft_sizes': [1024, 2048, 512],
                'hop_sizes': [120, 240, 50],
                'win_lengths': [600, 1200, 240],
                'window': 'hann_window'
            }

        self.stft_params = stft_params
        
        self.model = nn.ModuleDict()
        for i in range(len(stft_params['fft_sizes'])):
            self.model["disc_" + str(i)] = NLayerSpecDiscriminator(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                channels=channels,
                max_downsample_channels=max_downsample_channels,
                downsample_scales=downsample_scales,
            )

        # uses weight norm if self.use_weight_norm is true, else use spectral norm
        self.apply_norm()
        self.reset_parameters()

    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []
        i = 0
        for _, disc in self.model.items():
            # compute for real audio
            spec_real = stft(audio_real.squeeze(1), self.stft_params['fft_sizes'][i], self.stft_params['hop_sizes'][i],
                        self.stft_params['win_lengths'][i],
                        window=getattr(torch, self.stft_params['window'])(self.stft_params['win_lengths'][i])).transpose(1, 2).unsqueeze(1) # [B, 1, F, T]

            if self.spec_scale_pow != 0.0:
                spec_real = spec_real * torch.pow(spec_real.abs()+1e-6, self.spec_scale_pow)

            fmap_real = disc(spec_real)
            score_real = rearrange(fmap_real[-1], "B 1 T C -> B C T")
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)

            # compute for gen audio
            spec_gen = stft(audio_gen.squeeze(1), self.stft_params['fft_sizes'][i], self.stft_params['hop_sizes'][i],
                        self.stft_params['win_lengths'][i],
                        window=getattr(torch, self.stft_params['window'])(self.stft_params['win_lengths'][i])).transpose(1, 2).unsqueeze(1) # [B, 1, F, T]
            fmap_gen = disc(spec_gen)
            score_gen = rearrange(fmap_gen[-1], "B 1 T C -> B C T")
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

            i += 1

        return scores_real, scores_gen, fmaps_real, fmaps_gen

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def apply_norm(self):
        norm_fn = torch.nn.utils.weight_norm if self.use_weight_norm else torch.nn.utils.spectral_norm
        def _apply_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                norm_fn(m)
        self.apply(_apply_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
        self.apply(_reset_parameters)


class NLayerSpecDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_sizes=(5, 3),
                 channels=32,
                 max_downsample_channels=512,
                 downsample_scales=(2, 2, 2)):
        super().__init__()

        # check kernel size is valid
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.Conv2d(in_channels, channels,
                      kernel_size=kernel_sizes[0],
                      stride=2,
                      padding=kernel_sizes[0] // 2),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = channels
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)

            model[f"layer_{i + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=downsample_scale * 2 + 1,
                    stride=downsample_scale,
                    padding=downsample_scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        model[f"layer_{len(downsample_scales) + 1}"] = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=kernel_sizes[1],
                      padding=kernel_sizes[1] // 2),
            nn.LeakyReLU(0.2, True),
        )

        model[f"layer_{len(downsample_scales) + 2}"] = nn.Conv2d(
            out_chs, out_channels, kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2)

        self.model = model

    def forward(self, x):
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results

