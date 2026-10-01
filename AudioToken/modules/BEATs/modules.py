# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import warnings
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Allows you to scale grads in back pass
class GradMultiply(torch.autograd.Function):
    @staticmethod
    # In forward store scale and pass x;
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    # In back pass scale grads
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


"""A causal convolution is a convolution where the output at time t depends only on inputs at time t and before—not on any “future” inputs. This is essential in time-series or streaming scenarios (e.g., audio or language modeling) where you cannot peek ahead.
Concretely, for a 1D kernel of size 3:

Standard convolution (“same”):
You’d pad 1 zero on both left and right, so at position t the kernel spans inputs t–1,t,t+1.

Causal convolution:
You pad 2 zeros on the left and none on the right, so at output position t the kernel still spans three inputs, but now they are 
t–2,t–1,t—never peeking at t+1 or beyond.

This left-only padding plus right-end trimming is exactly what that remove = kernel_size–1 handles."""

class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x

# Swish Activation fn
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)

# A Gated Linear Unit. Learns both “values” and “gates” in one linear layer, then gates the values. Used as activation function
class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        # to be consistent with GLU_Linear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2])
        else:
            x = (x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2]))

        return x

# gelu_accurate: a slightly different mathematical approximation (from the original BERT paper) that some prefer for numeric stability.
def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


# gelu: wraps PyTorch’s built-in GELU.
def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


# During training, randomly zeros out (and rescales) blocks of weights to simulate quantization noise.
# Quantization noise is a training-time regularization technique that simulates the effect of reducing a model’s weight precision.
# e.g. from 32-bit floats down to 8-bit or even fewer bits—by randomly zeroing out (“dropping”) blocks of weights during the forward pass.
def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
            
    """A forward pre-hook runs just before the module’s own forward() method is called.
    In our case, _forward_pre_hook is done when you call module(input) (during training):
        Check if we’re in training mode.
        If so, randomly zero out blocks of the layer’s weight tensor (according to the quantization-noise scheme).
        Rescale the remaining weights to keep their expected magnitude unchanged.
    """
    module.register_forward_pre_hook(_forward_pre_hook)
    return module
