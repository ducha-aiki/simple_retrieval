# Vendored from https://github.com/HITCSC/CLIDD
# License: MIT
#
# Original authors: HITCSC
# Modules inlined: model/modules.py, model/triton_plugin.py, model/model.py, clidd.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# model/modules.py
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    """Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class ResNetLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim1, dim2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim2),
            nn.ReLU(True),
            nn.Conv2d(dim2, dim2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim2)
        )
        self.skip = nn.Identity() if dim1 == dim2 else nn.Conv2d(dim1, dim2, 1)

    def forward(self, x: torch.Tensor):
        return F.relu(self.layer(x) + self.skip(x))


class FasterNetLayer(nn.Module):
    def __init__(self, dim, n_div, n_mul):
        super().__init__()
        self.conv1 = nn.Conv2d(dim // n_div, dim // n_div, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * n_mul, 1, bias=False),
            nn.BatchNorm2d(dim * n_mul),
            nn.ReLU(True),
            nn.Conv2d(dim * n_mul, dim, 1)
        )
        self.dim1 = dim // n_div

    def forward(self, x: torch.Tensor):
        _x = torch.cat([
            self.conv1(x[:, :self.dim1]),
            x[:, self.dim1:]
        ], 1)
        return x + self.conv2(_x)


# ---------------------------------------------------------------------------
# model/triton_plugin.py
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    @triton.jit
    def deformable_sample_project_forward_kernel(
        output_ptr, input_ptr, grid_ptr, weight_ptr, bias_ptr,
        stride_out_b, stride_out_n, stride_out_c,
        stride_in_b, stride_in_h, stride_in_w, stride_in_c,
        stride_grid_b, stride_grid_n, stride_grid_m, stride_grid_d,
        stride_w_m, stride_w_cin, stride_w_cout,
        B, H, W, C_in, N, C_out,
        with_bias: tl.constexpr,
        align_corners: tl.constexpr,
        M: tl.constexpr,
        BLOCK_SIZE_C_OUT: tl.constexpr,
        BLOCK_SIZE_C_IN: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_n_blocks = tl.cdiv(N, BLOCK_N)
        b_idx = pid // num_n_blocks
        n_block_idx = pid % num_n_blocks

        n_start = n_block_idx * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N

        for c_out_start in range(0, C_out, BLOCK_SIZE_C_OUT):
            c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
            mask_c_out = c_out_offsets < C_out

            acc = tl.zeros((BLOCK_N, BLOCK_SIZE_C_OUT), dtype=tl.float32)

            for m in range(M):
                grid_offset = b_idx * stride_grid_b + n_offsets * stride_grid_n + m * stride_grid_m
                x = tl.load(grid_ptr + grid_offset, mask=mask_n, other=0.0)
                y = tl.load(grid_ptr + grid_offset + stride_grid_d, mask=mask_n, other=0.0)

                if align_corners:
                    ix = (x + 1) / 2.0 * (W - 1)
                    iy = (y + 1) / 2.0 * (H - 1)
                else:
                    ix = ((x + 1) * W - 1) / 2.0
                    iy = ((y + 1) * H - 1) / 2.0

                ix0 = tl.math.floor(ix).to(tl.int32)
                iy0 = tl.math.floor(iy).to(tl.int32)
                dx = ix - ix0
                dy = iy - iy0
                w00 = (1.0 - dx) * (1.0 - dy)
                w01 = (1.0 - dx) * dy
                w10 = dx * (1.0 - dy)
                w11 = dx * dy
                ix1 = ix0 + 1
                iy1 = iy0 + 1

                mask_x0 = (ix0 >= 0) & (ix0 < W)
                mask_x1 = (ix1 >= 0) & (ix1 < W)
                mask_y0 = (iy0 >= 0) & (iy0 < H)
                mask_y1 = (iy1 >= 0) & (iy1 < H)

                m_00 = mask_n[:, None] & mask_x0[:, None] & mask_y0[:, None]
                m_01 = mask_n[:, None] & mask_x0[:, None] & mask_y1[:, None]
                m_10 = mask_n[:, None] & mask_x1[:, None] & mask_y0[:, None]
                m_11 = mask_n[:, None] & mask_x1[:, None] & mask_y1[:, None]

                for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
                    c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
                    mask_c_in = c_in_offsets < C_in

                    ptr_base = input_ptr + b_idx * stride_in_b + c_in_offsets[None, :] * stride_in_c
                    ptr_00 = ptr_base + (iy0 * stride_in_h + ix0 * stride_in_w)[:, None]
                    ptr_01 = ptr_base + (iy1 * stride_in_h + ix0 * stride_in_w)[:, None]
                    ptr_10 = ptr_base + (iy0 * stride_in_h + ix1 * stride_in_w)[:, None]
                    ptr_11 = ptr_base + (iy1 * stride_in_h + ix1 * stride_in_w)[:, None]

                    val_00 = tl.load(ptr_00, mask=m_00 & mask_c_in[None, :], other=0.0)
                    val_01 = tl.load(ptr_01, mask=m_01 & mask_c_in[None, :], other=0.0)
                    val_10 = tl.load(ptr_10, mask=m_10 & mask_c_in[None, :], other=0.0)
                    val_11 = tl.load(ptr_11, mask=m_11 & mask_c_in[None, :], other=0.0)

                    s = w00[:, None] * val_00 + w01[:, None] * val_01 + w10[:, None] * val_10 + w11[:, None] * val_11

                    w_ptr = weight_ptr + m * stride_w_m + \
                            c_in_offsets[:, None] * stride_w_cin + \
                            c_out_offsets[None, :] * stride_w_cout
                    W_block = tl.load(w_ptr, mask=mask_c_in[:, None] & mask_c_out[None, :], other=0.0).to(tl.float32)

                    acc += tl.dot(s, W_block)

            if with_bias:
                bias_block = tl.load(bias_ptr + c_out_offsets, mask=mask_c_out, other=0.0)
                acc += bias_block[None, :]

            output_ptr_offset = output_ptr + b_idx * stride_out_b + \
                                n_offsets[:, None] * stride_out_n + \
                                c_out_offsets[None, :] * stride_out_c
            tl.store(output_ptr_offset, acc, mask=mask_n[:, None] & mask_c_out[None, :])

    @triton.jit
    def deformable_sample_project_backward_kernel(
        input_ptr, grid_ptr, weight_ptr, bias_ptr,
        grad_input_ptr, grad_grid_ptr, grad_weight_ptr, grad_bias_ptr,
        dout_ptr,
        stride_in_b, stride_in_h, stride_in_w, stride_in_c,
        stride_grid_b, stride_grid_n, stride_grid_m, stride_grid_d,
        stride_w_m, stride_w_cin, stride_w_cout,
        stride_grad_in_b, stride_grad_in_c, stride_grad_in_h, stride_grad_in_w,
        stride_grad_grid_b, stride_grad_grid_n, stride_grad_grid_m, stride_grad_grid_d,
        stride_grad_w_m, stride_grad_w_cin, stride_grad_w_cout,
        stride_dout_b, stride_dout_n, stride_dout_c,
        B, H, W, C_in, N, C_out,
        with_bias: tl.constexpr,
        align_corners: tl.constexpr,
        M: tl.constexpr,
        BLOCK_SIZE_C_OUT: tl.constexpr,
        BLOCK_SIZE_C_IN: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_n_blocks = tl.cdiv(N, BLOCK_N)
        b_idx = pid // num_n_blocks
        n_block_idx = pid % num_n_blocks

        n_start = n_block_idx * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N

        if with_bias:
            for c_out_start in range(0, C_out, BLOCK_SIZE_C_OUT):
                c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
                mask_c_out = c_out_offsets < C_out
                dout_offset = b_idx * stride_dout_b + n_offsets[:, None] * stride_dout_n + c_out_offsets[None, :] * stride_dout_c
                dout_block = tl.load(dout_ptr + dout_offset, mask=mask_n[:, None] & mask_c_out[None, :], other=0.0).to(tl.float32)
                d_bias = tl.sum(dout_block, axis=0)
                tl.atomic_add(grad_bias_ptr + c_out_offsets, d_bias, mask=mask_c_out)

        for m in range(M):
            grid_base = b_idx * stride_grid_b + n_offsets * stride_grid_n + m * stride_grid_m
            x = tl.load(grid_ptr + grid_base, mask=mask_n, other=0.0)
            y = tl.load(grid_ptr + grid_base + stride_grid_d, mask=mask_n, other=0.0)

            if align_corners:
                ix = (x + 1.0) / 2.0 * (W - 1)
                iy = (y + 1.0) / 2.0 * (H - 1)
                scale_x = (W - 1) / 2.0
                scale_y = (H - 1) / 2.0
            else:
                ix = ((x + 1.0) * W - 1.0) / 2.0
                iy = ((y + 1.0) * H - 1.0) / 2.0
                scale_x = W / 2.0
                scale_y = H / 2.0

            ix0 = tl.math.floor(ix).to(tl.int32)
            iy0 = tl.math.floor(iy).to(tl.int32)
            dx = ix - ix0
            dy = iy - iy0
            w00 = (1.0 - dx) * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w10 = dx * (1.0 - dy)
            w11 = dx * dy
            ix1 = ix0 + 1
            iy1 = iy0 + 1

            mask_x0 = (ix0 >= 0) & (ix0 < W)
            mask_x1 = (ix1 >= 0) & (ix1 < W)
            mask_y0 = (iy0 >= 0) & (iy0 < H)
            mask_y1 = (iy1 >= 0) & (iy1 < H)

            m_00 = mask_n[:, None] & mask_x0[:, None] & mask_y0[:, None]
            m_01 = mask_n[:, None] & mask_x0[:, None] & mask_y1[:, None]
            m_10 = mask_n[:, None] & mask_x1[:, None] & mask_y0[:, None]
            m_11 = mask_n[:, None] & mask_x1[:, None] & mask_y1[:, None]

            dL_dix_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            dL_diy_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
                c_in_offsets = c_in_start + tl.arange(0, BLOCK_SIZE_C_IN)
                mask_c_in = c_in_offsets < C_in

                ptr_base = input_ptr + b_idx * stride_in_b + c_in_offsets[None, :] * stride_in_c
                ptr_00 = ptr_base + (iy0 * stride_in_h + ix0 * stride_in_w)[:, None]
                ptr_01 = ptr_base + (iy1 * stride_in_h + ix0 * stride_in_w)[:, None]
                ptr_10 = ptr_base + (iy0 * stride_in_h + ix1 * stride_in_w)[:, None]
                ptr_11 = ptr_base + (iy1 * stride_in_h + ix1 * stride_in_w)[:, None]

                val_00 = tl.load(ptr_00, mask=m_00 & mask_c_in[None, :], other=0.0)
                val_01 = tl.load(ptr_01, mask=m_01 & mask_c_in[None, :], other=0.0)
                val_10 = tl.load(ptr_10, mask=m_10 & mask_c_in[None, :], other=0.0)
                val_11 = tl.load(ptr_11, mask=m_11 & mask_c_in[None, :], other=0.0)

                s = w00[:, None] * val_00 + w01[:, None] * val_01 + w10[:, None] * val_10 + w11[:, None] * val_11

                g_s = tl.zeros((BLOCK_N, BLOCK_SIZE_C_IN), dtype=tl.float32)

                for c_out_start in range(0, C_out, BLOCK_SIZE_C_OUT):
                    c_out_offsets = c_out_start + tl.arange(0, BLOCK_SIZE_C_OUT)
                    mask_c_out = c_out_offsets < C_out

                    dout_offset = b_idx * stride_dout_b + n_offsets[:, None] * stride_dout_n + c_out_offsets[None, :] * stride_dout_c
                    dout_block = tl.load(dout_ptr + dout_offset, mask=mask_n[:, None] & mask_c_out[None, :], other=0.0).to(tl.float32)

                    w_ptr = weight_ptr + m * stride_w_m + c_in_offsets[:, None] * stride_w_cin + c_out_offsets[None, :] * stride_w_cout
                    W_block = tl.load(w_ptr, mask=mask_c_in[:, None] & mask_c_out[None, :], other=0.0).to(tl.float32)

                    g_s += tl.dot(dout_block, tl.trans(W_block))

                    contrib_w = tl.dot(tl.trans(s), dout_block)
                    grad_w_ptr = grad_weight_ptr + m * stride_grad_w_m + c_in_offsets[:, None] * stride_grad_w_cin + c_out_offsets[None, :] * stride_grad_w_cout
                    tl.atomic_add(grad_w_ptr, contrib_w, mask=mask_c_in[:, None] & mask_c_out[None, :])

                contrib_00 = w00[:, None] * g_s
                contrib_01 = w01[:, None] * g_s
                contrib_10 = w10[:, None] * g_s
                contrib_11 = w11[:, None] * g_s

                grad_ptr_base = grad_input_ptr + b_idx * stride_grad_in_b + c_in_offsets[None, :] * stride_grad_in_c
                grad_ptr_00 = grad_ptr_base + (iy0 * stride_grad_in_h + ix0 * stride_grad_in_w)[:, None]
                grad_ptr_01 = grad_ptr_base + (iy1 * stride_grad_in_h + ix0 * stride_grad_in_w)[:, None]
                grad_ptr_10 = grad_ptr_base + (iy0 * stride_grad_in_h + ix1 * stride_grad_in_w)[:, None]
                grad_ptr_11 = grad_ptr_base + (iy1 * stride_grad_in_h + ix1 * stride_grad_in_w)[:, None]

                tl.atomic_add(grad_ptr_00, contrib_00, mask=m_00 & mask_c_in[None, :])
                tl.atomic_add(grad_ptr_01, contrib_01, mask=m_01 & mask_c_in[None, :])
                tl.atomic_add(grad_ptr_10, contrib_10, mask=m_10 & mask_c_in[None, :])
                tl.atomic_add(grad_ptr_11, contrib_11, mask=m_11 & mask_c_in[None, :])

                dwdix_00 = -(1.0 - dy)
                dwdix_01 = -dy
                dwdix_10 = (1.0 - dy)
                dwdix_11 = dy
                dwdiy_00 = -(1.0 - dx)
                dwdiy_01 = (1.0 - dx)
                dwdiy_10 = -dx
                dwdiy_11 = dx

                ds_dix = dwdix_00[:, None] * val_00 + dwdix_01[:, None] * val_01 + dwdix_10[:, None] * val_10 + dwdix_11[:, None] * val_11
                ds_diy = dwdiy_00[:, None] * val_00 + dwdiy_01[:, None] * val_01 + dwdiy_10[:, None] * val_10 + dwdiy_11[:, None] * val_11

                dL_dix_acc += tl.sum(g_s * ds_dix, axis=1)
                dL_diy_acc += tl.sum(g_s * ds_diy, axis=1)

            dL_dx_norm = dL_dix_acc * scale_x
            dL_dy_norm = dL_diy_acc * scale_y

            grad_grid_x_ptr = grad_grid_ptr + b_idx * stride_grad_grid_b + n_offsets * stride_grad_grid_n + m * stride_grad_grid_m
            grad_grid_y_ptr = grad_grid_x_ptr + stride_grad_grid_d
            tl.store(grad_grid_x_ptr, dL_dx_norm, mask=mask_n)
            tl.store(grad_grid_y_ptr, dL_dy_norm, mask=mask_n)

    def next_pow2_ge(a: int) -> int:
        if a <= 1:
            return 1
        return 1 << ((a - 1).bit_length())

    class DeformableSampleProject(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input: torch.Tensor, grid: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, is_input_nhwc: bool = False, align_corners: bool = False):
            input_nhwc = input.permute(0, 2, 3, 1).contiguous() if not is_input_nhwc else input
            ctx.save_for_backward(input_nhwc, grid, weight, bias)
            ctx.is_input_nhwc = bool(is_input_nhwc)
            ctx.align_corners = bool(align_corners)

            assert input_nhwc.is_cuda and grid.is_cuda and weight.is_cuda
            assert bias is None or bias.is_cuda
            B, H, W, C_in = input_nhwc.shape
            _B, N, M, _2 = grid.shape
            C_out, _C_in, _, _M = weight.shape
            assert C_in == _C_in
            assert M == _M
            assert B == _B

            with_bias = bias is not None
            if bias is None:
                bias = input_nhwc.new_zeros((C_out,), dtype=input_nhwc.dtype)
            weight_reshaped = weight[:, :, 0].permute(2, 1, 0).contiguous()

            output = torch.empty((B, N, C_out), device=input_nhwc.device, dtype=input_nhwc.dtype)
            ostride_b = output.stride(0)
            ostride_n = output.stride(1)
            ostride_c = output.stride(2)

            BLOCK_N = 32
            BLOCK_SIZE_C_IN = max(min(32, next_pow2_ge(C_in)), 16)
            BLOCK_SIZE_C_OUT = max(min(128, next_pow2_ge(C_out)), 16)

            grid_launch = (triton.cdiv(N, BLOCK_N) * B,)

            deformable_sample_project_forward_kernel[grid_launch](
                output, input_nhwc, grid, weight_reshaped, bias,
                ostride_b, ostride_n, ostride_c,
                input_nhwc.stride(0), input_nhwc.stride(1), input_nhwc.stride(2), input_nhwc.stride(3),
                grid.stride(0), grid.stride(1), grid.stride(2), grid.stride(3),
                weight_reshaped.stride(0), weight_reshaped.stride(1), weight_reshaped.stride(2),
                B, H, W, C_in, N, C_out,
                with_bias=with_bias,
                align_corners=align_corners,
                M=M,
                BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
                BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input_nhwc, grid, weight, bias = ctx.saved_tensors
            is_input_nhwc = ctx.is_input_nhwc
            align_corners = ctx.align_corners

            assert input_nhwc.is_cuda and grid.is_cuda and weight.is_cuda and grad_output.is_cuda
            assert bias is None or bias.is_cuda
            B, H, W, C_in = input_nhwc.shape
            _B, N, M, _2 = grid.shape
            C_out, _C_in, _, _M = weight.shape
            assert C_in == _C_in
            assert M == _M
            assert grad_output.shape == (B, N, C_out)
            gostride_b = grad_output.stride(0)
            gostride_n = grad_output.stride(1)
            gostride_c = grad_output.stride(2)

            with_bias = bias is not None
            if bias is None:
                bias = input_nhwc.new_zeros((C_out,), dtype=input_nhwc.dtype)
            weight_reshaped = weight[:, :, 0].permute(2, 1, 0).contiguous()

            grad_input_nhwc = torch.zeros_like(input_nhwc)
            grad_grid = torch.zeros_like(grid)
            grad_weight_reshaped = torch.zeros_like(weight_reshaped)
            grad_bias = torch.zeros_like(bias) if with_bias else None

            BLOCK_N = 32
            BLOCK_SIZE_C_IN = max(min(32, next_pow2_ge(C_in)), 16)
            BLOCK_SIZE_C_OUT = max(min(128, next_pow2_ge(C_out)), 16)
            grid_launch = (triton.cdiv(N, BLOCK_N) * B,)

            deformable_sample_project_backward_kernel[grid_launch](
                input_nhwc, grid, weight_reshaped, bias,
                grad_input_nhwc, grad_grid, grad_weight_reshaped, grad_bias,
                grad_output,
                input_nhwc.stride(0), input_nhwc.stride(1), input_nhwc.stride(2), input_nhwc.stride(3),
                grid.stride(0), grid.stride(1), grid.stride(2), grid.stride(3),
                weight_reshaped.stride(0), weight_reshaped.stride(1), weight_reshaped.stride(2),
                grad_input_nhwc.stride(0), grad_input_nhwc.stride(3), grad_input_nhwc.stride(1), grad_input_nhwc.stride(2),
                grad_grid.stride(0), grad_grid.stride(1), grad_grid.stride(2), grad_grid.stride(3),
                grad_weight_reshaped.stride(0), grad_weight_reshaped.stride(1), grad_weight_reshaped.stride(2),
                gostride_b, gostride_n, gostride_c,
                B, H, W, C_in, N, C_out,
                with_bias=with_bias,
                align_corners=align_corners,
                M=M,
                BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
                BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
            grad_input = grad_input_nhwc.permute(0, 3, 1, 2).contiguous() if not is_input_nhwc else grad_input_nhwc
            grad_weight = grad_weight_reshaped.permute(2, 1, 0).unsqueeze(2).contiguous()
            return grad_input, grad_grid, grad_weight, grad_bias, None, None, None

    def deformable_sample_project(input, grid, weight, bias, *, is_input_nhwc: bool = False, align_corners: bool = False):
        return DeformableSampleProject.apply(input, grid, weight, bias, is_input_nhwc, align_corners)

    _TRITON_AVAILABLE = True

except ImportError:
    _TRITON_AVAILABLE = False

    def deformable_sample_project(input, grid, weight, bias, *, is_input_nhwc: bool = False, align_corners: bool = False):
        raise RuntimeError(
            "triton is required for CLIDD on CUDA. Install with: pip install triton"
        )


# ---------------------------------------------------------------------------
# model/model.py
# ---------------------------------------------------------------------------

class _CLIDDModel(nn.Module):

    def __init__(self, c1, c2, c3, r1, r2, r3, cdesc, cdetect, M, max_offset):
        super().__init__()

        csum = c1 + c2 + c3
        self.max_offset = max_offset

        self.block1 = nn.Sequential(
            BasicLayer(3, c1, 4, 2, 1),
            BasicLayer(c1, c1),
            *[ResNetLayer(c1, c1) for _ in range(r1)],
        )
        self.block2 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            *[ResNetLayer(c1, c2) if _ == 0 else ResNetLayer(c2, c2) for _ in range(r2)],
        )
        self.block3 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            *[ResNetLayer(c2, c3) if _ == 0 else ResNetLayer(c3, c3) for _ in range(r3)],
        )

        self.conv1 = nn.Conv2d(c1, cdetect, 1)
        self.conv2 = nn.Conv2d(c2, cdetect, 1)
        self.conv3 = nn.Conv2d(c3, cdetect, 1)

        self.desc_head = nn.Identity()
        self.score_head = nn.Sequential(
            nn.Conv2d(cdetect, cdetect, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cdetect, 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        self.dcn = nn.Conv2d(csum, M * 2 * 3, 1)
        self.post_conv = nn.Conv2d(csum, cdesc, (1, M))

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        desc = (x1, x2, x3)

        feat = self.conv2(x2) + F.interpolate(self.conv3(x3), scale_factor=4, mode='bilinear', align_corners=False)
        feat = self.conv1(x1) + F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)

        score = self.score_head(feat)

        return desc, score

    def sample(self, dense, kpts, *, align_corners=False):
        H, W = dense[0].shape[-2:]
        H = H * 2
        W = W * 2
        c0 = dense[0].shape[1]
        c1 = dense[1].shape[1]
        c2 = dense[2].shape[1]

        weight_offset1, weight_offset2, weight_offset3 = self.dcn.weight.split([c0, c1, c2], 1)
        bias_offset = self.dcn.bias
        weight_post1, weight_post2, weight_post3 = self.post_conv.weight.split([c0, c1, c2], 1)
        bias_post = self.post_conv.bias

        dense[0] = dense[0].permute(0, 2, 3, 1).contiguous()
        dense[1] = dense[1].permute(0, 2, 3, 1).contiguous()
        dense[2] = dense[2].permute(0, 2, 3, 1).contiguous()
        is_input_nhwc = True

        offset = deformable_sample_project(dense[0], kpts, weight_offset1, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
                 deformable_sample_project(dense[1], kpts, weight_offset2, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
                 deformable_sample_project(dense[2], kpts, weight_offset3, bias_offset, is_input_nhwc=is_input_nhwc, align_corners=align_corners)

        offset = offset.unflatten(-1, (-1, 2)).clamp(-self.max_offset, self.max_offset)
        offset_1, offset_2, offset_3 = (offset / torch.tensor([W, H]).to(kpts) * 2).chunk(3, -2)
        kpts = kpts.to(offset)

        desc = deformable_sample_project(dense[0], kpts + offset_1, weight_post1, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
               deformable_sample_project(dense[1], kpts + offset_2, weight_post2, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
               deformable_sample_project(dense[2], kpts + offset_3, weight_post3, bias_post, is_input_nhwc=is_input_nhwc, align_corners=align_corners)
        return F.normalize(desc, 2, -1)


# ---------------------------------------------------------------------------
# clidd.py  (CLIDD wrapper)
# ---------------------------------------------------------------------------

# Default weights directory: <repo_root>/weights/
_DEFAULT_WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights'))


class CLIDD(nn.Module):

    cfgs = {
        'A48':  {'c1': 4,  'c2': 4,   'c3': 4,   'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 48,  'cdetect': 4, 'M': 4,  'max_offset': 128},
        'N64':  {'c1': 8,  'c2': 8,   'c3': 8,   'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64,  'cdetect': 8, 'M': 8,  'max_offset': 128},
        'T64':  {'c1': 8,  'c2': 16,  'c3': 24,  'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64,  'cdetect': 8, 'M': 8,  'max_offset': 128},
        'S64':  {'c1': 8,  'c2': 24,  'c3': 32,  'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64,  'cdetect': 8, 'M': 16, 'max_offset': 128},
        'M64':  {'c1': 16, 'c2': 32,  'c3': 48,  'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64,  'cdetect': 8, 'M': 16, 'max_offset': 128},
        'L64':  {'c1': 16, 'c2': 48,  'c3': 96,  'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64,  'cdetect': 8, 'M': 16, 'max_offset': 128},
        'G128': {'c1': 16, 'c2': 64,  'c3': 256, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
        'E128': {'c1': 16, 'c2': 64,  'c3': 256, 'r1': 1, 'r2': 2, 'r3': 2, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
        'U128': {'c1': 32, 'c2': 128, 'c3': 256, 'r1': 1, 'r2': 2, 'r3': 2, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
    }

    def __init__(self, cfg, top_k, radius=2, score=-5, weights_path=None):
        """
        Args:
            cfg: Model variant, e.g. 'E128'.
            top_k: Number of keypoints to keep (or None for all).
            radius: NMS radius.
            score: Score threshold.
            weights_path: Path to the .pth weight file. Defaults to
                          <repo_root>/weights/{cfg}.pth.
        """
        super().__init__()
        assert top_k is None or top_k > 0
        self.top_k = top_k
        self.radius = radius
        self.score_thresh = score

        self.model = _CLIDDModel(**self.cfgs[cfg])

        if weights_path is None:
            weights_path = os.path.join(_DEFAULT_WEIGHTS_DIR, f'{cfg}.pth')
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))

        if radius > 0:
            self.mp = nn.MaxPool2d(radius * 2 + 1, 1, radius)

    @torch.inference_mode()
    def forward(self, x):
        B, _, oH, oW = x.shape
        nH = oH // 32 * 32
        nW = oW // 32 * 32
        size = torch.tensor([nW, nH], dtype=x.dtype, device=x.device)
        scale = torch.tensor([oW / nW, oH / nH], dtype=x.dtype, device=x.device)
        if oW != nW or oH != nH:
            x = F.interpolate(x, (nH, nW), mode='bilinear', align_corners=True)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        raw_desc, raw_detect = self.model(x)

        if self.radius > 0:
            detect1 = raw_detect == self.mp(raw_detect)
        else:
            detect1 = torch.ones_like(raw_detect, dtype=torch.bool)
        detect1[..., :, :4] = False
        detect1[..., :, -4:] = False
        detect1[..., :4, :] = False
        detect1[..., -4:, :] = False

        detect2 = raw_detect > self.score_thresh
        detect = torch.logical_and(detect1, detect2)[:, 0]
        H = torch.arange(detect.shape[-2], dtype=x.dtype, device=x.device)
        W = torch.arange(detect.shape[-1], dtype=x.dtype, device=x.device)
        H, W = torch.meshgrid(H, W)
        ind = torch.stack([W, H], dim=-1)
        kpts = [ind[detect[b]] for b in range(B)]
        scores = [raw_detect[b, 0, detect[b]] for b in range(B)]

        if self.top_k is not None:
            for i in range(B):
                score, idx = scores[i].topk(min(self.top_k, scores[i].shape[0]))
                scores[i] = score
                kpts[i] = kpts[i][idx]

        descs = [
            self.model.sample(
                [r[b:b + 1] for r in raw_desc],
                (kpts[b] + 0.5).reshape(1, -1, 1, 2) / size * 2 - 1
            )[0] if kpts[b].shape[0] > 0 else raw_detect.new_zeros([0, 0])
            for b in range(B)
        ]

        return [
            {'keypoints': kpts[b] * scale,
             'scores': scores[b],
             'descriptors': descs[b]}
            for b in range(B)
        ]

    def match(self, desc0: torch.Tensor, desc1: torch.Tensor, beta=20):
        if desc0.shape[0] == 0 or desc1.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        dist: torch.Tensor = desc0 @ desc1.t()
        dist.sub_(1).multiply_(beta).exp_()
        sum1 = dist.sum(dim=-1, keepdim=True)
        sum2 = dist.sum(dim=-2, keepdim=True)
        dist = dist.square_().div_(sum1).div_(sum2)

        nn12 = torch.max(dist, dim=1)[1]
        nn21 = torch.max(dist, dim=0)[1]
        ids1 = torch.arange(0, dist.shape[0], device=dist.device)
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]])

        dist = dist[ids1[mask], nn12[mask]]
        mask = dist > 0.01
        matches = matches[:, mask]
        idxs1 = matches[0].data.cpu().numpy()
        idxs2 = matches[1].data.cpu().numpy()
        return idxs1, idxs2
