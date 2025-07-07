import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import MaskedConv2d

try:
    from mamba_ssm import Mamba
except ImportError:
    print("FATAL ERROR: mamba_ssm is not installed. Please ensure your environment is set up correctly.")
    exit()


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        assert C == self.d_model, f"Input channels {C} do not match d_model {self.d_model}"
        x_reshaped = x.flatten(2).transpose(1, 2)
        mamba_out = self.mamba(self.norm1(x_reshaped))
        mamba_out_norm = self.norm2(mamba_out)
        out_reshaped = mamba_out_norm.transpose(1, 2).view(B, C, H, W)
        return identity + out_reshaped


class MyHierarchicalEncoder(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), MambaBlock(N), MambaBlock(N),
            conv(N, N, kernel_size=5, stride=2), MambaBlock(N), MambaBlock(N),
            conv(N, N, kernel_size=5, stride=2), MambaBlock(N), MambaBlock(N),
            conv(N, M, kernel_size=5, stride=2),
        )

    def forward(self, x):
        return self.g_a(x)


class MyHierarchicalDecoder(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2), MambaBlock(N), MambaBlock(N),
            deconv(N, N, kernel_size=5, stride=2), MambaBlock(N), MambaBlock(N),
            deconv(N, N, kernel_size=5, stride=2), MambaBlock(N), MambaBlock(N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x):
        return self.g_s(x)


class Hyperprior(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3), nn.GELU(),
            conv(N, N, stride=2, kernel_size=5), nn.GELU(),
            conv(N, N, stride=2, kernel_size=5),
        )
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5), nn.GELU(),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5), nn.GELU(),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, y):
        z = self.h_a(torch.abs(y))
        return z, self.h_s(z)


class ContextModel(nn.Module):
    def __init__(self, M, N):
        super().__init__()
        self.context_prediction = nn.Sequential(
            MaskedConv2d(M, N, kernel_size=5, padding=2, stride=1), nn.GELU(),
            MaskedConv2d(N, N, kernel_size=5, padding=2, stride=1), nn.GELU(),
            MaskedConv2d(N, M * 2, kernel_size=5, padding=2, stride=1),
        )

    def forward(self, y_hat):
        return self.context_prediction(y_hat)


class MySOTAMambaModel(CompressionModel):
    def __init__(self, N=192, M=320):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = N
        self.M = M
        self.g_a = MyHierarchicalEncoder(N, M)
        self.g_s = MyHierarchicalDecoder(N, M)
        self.hyperprior = Hyperprior(N, M)
        self.context_model = ContextModel(M, N)
        self.gaussian_conditional = GaussianConditional(None)

    def quantize(self, inputs, mode):
        if mode == "noise":
            noise = torch.empty_like(inputs).uniform_(-0.5, 0.5)
            return inputs + noise
        elif mode == "round":
            return torch.round(inputs)
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

    def forward(self, x):
        y = self.g_a(x)
        z, hyper_params = self.hyperprior(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        hyper_scales_hat, hyper_means_hat = hyper_params.chunk(2, 1)

        y_hat = self.quantize(y, "noise" if self.training else "round")
        context_params = self.context_model(y_hat)
        context_scales, context_means = context_params.chunk(2, 1)

        final_scales = torch.clamp(hyper_scales_hat + context_scales, min=1e-3)
        final_means = torch.clamp(hyper_means_hat + context_means, min=-1e3, max=1e3)

        _, y_likelihoods = self.gaussian_conditional(y, final_scales, means=final_means)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def compress(self, x):
        y = self.g_a(x)
        z, _ = self.hyperprior(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.hyperprior.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        context_params = self.context_model(self.quantize(y, "round"))
        context_scales, context_means = context_params.chunk(2, 1)

        final_scales = torch.clamp(hyper_scales + context_scales, min=1e-3)
        final_means = torch.clamp(hyper_means + context_means, min=-1e3, max=1e3)

        indexes = self.gaussian_conditional.build_indexes(final_scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=final_means)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_params = self.hyperprior.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        hyper_scales = torch.clamp(hyper_scales, min=1e-3)
        hyper_means = torch.clamp(hyper_means, min=-1e3, max=1e3)

        B, _, H, W = hyper_scales.shape
        y_hat = torch.zeros((B, self.M, H, W), device=z_hat.device)

        for i in range(H):
            for j in range(W):
                context_params = self.context_model(y_hat)
                context_scales, context_means = context_params.chunk(2, 1)

                final_scales = torch.clamp(hyper_scales + context_scales, min=1e-3)
                final_means = torch.clamp(hyper_means + context_means, min=-1e3, max=1e3)

                indexes = self.gaussian_conditional.build_indexes(final_scales)
                y_slice = self.gaussian_conditional.decompress(
                    strings[0],
                    indexes[:, :, i:i+1, j:j+1],
                    means=final_means[:, :, i:i+1, j:j+1]
                )
                y_hat[:, :, i, j] = y_slice.squeeze()

        x_hat = self.g_s(y_hat)
        x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
        return {"x_hat": x_hat}
