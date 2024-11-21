# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_varlen_kvpacked_func

from torch.nn.functional import scaled_dot_product_attention as sdpa


def flops(batch, seqlen, headdim, nheads, mask_ratio, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim * mask_ratio
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.bfloat16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
mask_types = ["causal", "two_range"]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = ["Flash+", "FlashCausal", "PytorchSDPA"]

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for mask_type in mask_types:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (mask_type, headdim, batch_size, seqlen)
            nheads = dim // headdim
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            q = qkv[:, :, 0, :, :].squeeze().transpose(1, 2)
            k = qkv[:, :, 1, :, :].squeeze().transpose(1, 2)
            v = qkv[:, :, 2, :, :].squeeze().transpose(1, 2)
            q_unpad = qkv[:, :, 0, :, :].reshape(batch_size * seqlen, nheads, headdim)
            kv_unpad = qkv[:, :, 1:, :, :].reshape(batch_size * seqlen, 2, nheads, headdim)
            cu_seqlens_q = F.pad(
                torch.cumsum(
                    torch.full((batch_size,), seqlen, device=device, dtype=torch.int32), dim=0
                ), (1, 0)
            ).to(torch.int32)
            cu_seqlens_kv = cu_seqlens_q.clone()
            if mask_type == "causal":
                attn_range = torch.empty((2, cu_seqlens_q[-1].item()), device=device, dtype=torch.int32)
                for b in range(batch_size):
                    for i in range(seqlen):
                        attn_range[0, cu_seqlens_q[b].item() + i] = 0
                        attn_range[1, cu_seqlens_q[b].item() + i] = i + 1
                attn_mask = torch.tril(
                    torch.ones(
                        batch_size,
                        1,
                        seqlen,
                        seqlen,
                        device=device,
                        dtype=bool,
                    )
                )
            elif mask_type == "two_range":
                attn_range = torch.empty((2, 2, cu_seqlens_q[-1].item()), device=device, dtype=torch.int32)
                for b in range(batch_size):
                    for i in range(seqlen):
                        # first range: 0 -> 128
                        attn_range[0, 0, cu_seqlens_q[b].item() + i] = 0
                        attn_range[0, 1, cu_seqlens_q[b].item() + i] = min(i + 1, 128)
                        # second range: i - 128 -> i
                        attn_range[1, 0, cu_seqlens_q[b].item() + i] = max(i - 128, 0)
                        attn_range[1, 1, cu_seqlens_q[b].item() + i] = i + 1
                attn_mask = torch.full(
                    (batch_size, 1, seqlen, seqlen), False, device=device, dtype=bool
                )
                for b in range(batch_size):
                    for i in range(seqlen):
                        attn_mask[b, 0, i, 0: min(i + 1, 128)] = True
                        attn_mask[b, 0, i, max(i - 128, 0): i + 1] = True

            f, b = time_fwd_bwd(
                flash_attn_varlen_kvpacked_func, q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_kv, seqlen, seqlen, dropout_p,
                attn_range=attn_range,
                repeats=repeats, verbose=False
            )
            time_f[config, "Flash+"] = f
            time_b[config, "Flash+"] = b

            try:
                qkv = qkv.detach().requires_grad_(True)
                f, b = time_fwd_bwd(
                    sdpa, q, k, v, attn_mask, repeats=repeats, verbose=False
                )
            except Exception as e:
                print(f"PytorchSDPA failed with error: {e}")
                f, b = float('nan'), float('nan')
            time_f[config, "PytorchSDPA"] = f
            time_b[config, "PytorchSDPA"] = b

            if mask_type == "causal":
                f, b = time_fwd_bwd(
                    flash_attn_varlen_kvpacked_func, q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_kv, seqlen, seqlen, dropout_p,
                    causal=True, force_split_kv=True,
                    repeats=repeats, verbose=False
                )
                time_f[config, "FlashCausal"] = f
                time_b[config, "FlashCausal"] = b
            else:
                time_f[config, "FlashCausal"] = float('nan')
                time_b[config, "FlashCausal"] = float('nan')

            if mask_type == "causal":
                mask_ratio = 0.5
            elif mask_type == "two_range":
                mask_ratio = 0
                for i in range(seqlen):
                    mask_ratio += min(i+1, 256)
                mask_ratio /= seqlen**2

            print(f"### mask_type={mask_type}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, mask_ratio, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, mask_ratio, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, mask_ratio, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                )

with open('flash2_attn_time.plk', 'wb') as fp:
    pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
