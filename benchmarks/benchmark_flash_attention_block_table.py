# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_varlen_kvpacked_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs, verbose=False)
    return time_f[1].mean

def _blockize_qkv_out(
    q_unpad: torch.Tensor,
    kv_unpad: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
):
    # reverse cumsum to get the original seqlens
    q_seqlens_orig = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_seqlens_orig = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    n_blocks_per_seq_q = (q_seqlens_orig + block_size_q - 1) // block_size_q
    n_blocks_per_seq_k = (k_seqlens_orig + block_size_k - 1) // block_size_k
    
    max_num_blocks_per_seq_q = n_blocks_per_seq_q.max()
    max_num_blocks_per_seq_k = n_blocks_per_seq_k.max()
    sum_num_blocks_q = n_blocks_per_seq_q.sum()
    sum_num_blocks_k = n_blocks_per_seq_k.sum()
    q_cache_size = sum_num_blocks_q * 2
    kv_cache_size = sum_num_blocks_k * 2
    # generate block table
    q_block_mapping = torch.tensor(np.random.choice(q_cache_size.item(), sum_num_blocks_q.item(), replace=False), device=q_unpad.device, dtype=torch.int32)
    cumsum_num_blocks_q = torch.concat([torch.zeros(1, device=n_blocks_per_seq_q.device, dtype=n_blocks_per_seq_q.dtype),
                                        n_blocks_per_seq_q.cumsum(0)], 0)
    q_block_table = [
        q_block_mapping[cumsum_num_blocks_q[i]:cumsum_num_blocks_q[i + 1]]
        for i in range(len(n_blocks_per_seq_q))
    ]

    kv_block_mapping = torch.tensor(np.random.choice(kv_cache_size.item(), sum_num_blocks_k.item(), replace=False), device=kv_unpad.device, dtype=torch.int32)
    cumsum_num_blocks_k = torch.concat([torch.zeros(1, device=n_blocks_per_seq_k.device, dtype=n_blocks_per_seq_k.dtype),
                                        n_blocks_per_seq_k.cumsum(0)], 0)
    kv_block_table = [
        kv_block_mapping[cumsum_num_blocks_k[i]:cumsum_num_blocks_k[i + 1]]
        for i in range(len(n_blocks_per_seq_k))
    ]

    out_block_mapping = torch.tensor(np.random.choice(q_cache_size.item(), sum_num_blocks_q.item(), replace=False), device=q_unpad.device, dtype=torch.int32)
    out_block_table = [
        out_block_mapping[cumsum_num_blocks_q[i]:cumsum_num_blocks_q[i + 1]]
        for i in range(len(n_blocks_per_seq_q))
    ]

    # create and fill in the q and kv cache
    q_cache = torch.zeros(q_cache_size, block_size_q, q_unpad.size(-2), q_unpad.size(-1), device=q_unpad.device, dtype=q_unpad.dtype)
    for i, blocks in enumerate(q_block_table):
        q_unpad_offset = cu_seqlens_q[i]
        q_unpad_max = cu_seqlens_q[i + 1]
        for block_idx, block_offset in enumerate(blocks):
            curr_q_offset = q_unpad_offset + block_idx * block_size_q
            curr_block_size = min(block_size_q, q_unpad_max - curr_q_offset)
            q_cache[block_offset, :curr_block_size] = q_unpad[curr_q_offset:curr_q_offset + curr_block_size]
    kv_cache = torch.zeros(kv_cache_size, block_size_k, 2, kv_unpad.size(-2), kv_unpad.size(-1), device=kv_unpad.device, dtype=kv_unpad.dtype)
    for i, blocks in enumerate(kv_block_table):
        kv_unpad_offset = cu_seqlens_k[i]
        kv_unpad_max = cu_seqlens_k[i + 1]
        for block_idx, block_offset in enumerate(blocks):
            curr_kv_offset = kv_unpad_offset + block_idx * block_size_k
            curr_block_size = min(block_size_k, kv_unpad_max - curr_kv_offset)
            kv_cache[block_offset, :curr_block_size] = kv_unpad[curr_kv_offset:curr_kv_offset + curr_block_size]
    # dont need to fill out caches
    out_dtype = q_unpad.dtype
    out_cache = torch.zeros(q_cache_size, block_size_q, q_unpad.size(-2), q_unpad.size(-1), device=q_unpad.device, dtype=out_dtype)
    lse_cache = torch.zeros(q_unpad.size(-2), q_cache_size, block_size_q, device=q_unpad.device, dtype=torch.float)

    # pad block table to max_num_blocks_per_seq
    q_block_table = torch.stack([torch.cat([block, torch.zeros(max_num_blocks_per_seq_q - len(block), device=block.device, dtype=block.dtype)])
                     for block in q_block_table])
    kv_block_table = torch.stack([torch.cat([block, torch.zeros(max_num_blocks_per_seq_k - len(block), device=block.device, dtype=block.dtype)])
                        for block in kv_block_table])
    out_block_table = torch.stack([torch.cat([block, torch.zeros(max_num_blocks_per_seq_q - len(block), device=block.device, dtype=block.dtype)])
                        for block in out_block_table])
    return q_block_table, kv_block_table, out_block_table, q_cache, kv_cache, out_cache, lse_cache


repeats = 30
device = 'cuda'
dtype = torch.bfloat16

bs_seqlen_vals = [(32, 1024), (16, 2048), (8, 4096), (4, 8192), (2, 16384), (1, 32768)]
block_sizes = [256, 512, 1024]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 1024
dropout_p = 0.0

methods = (["Flash2", "Flash2-Paged-QKVO", "Flash2-Paged-KV"]
           + (["Triton"] if attention_triton is not None else [])
           + (["xformers.c"] if xops is not None else [])
           + (["xformers.f"] if xops is not None else []))

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            for block_size in block_sizes:
                config = (causal, headdim, batch_size, seqlen, block_size)
                nheads = dim // headdim
                seqlens = torch.full((batch_size,), seqlen, device=device, dtype=torch.int32)
                total_seqlen = sum(seqlens).item()
                qkv = torch.randn(total_seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
                q = qkv[:, 0]
                kv = qkv[:, 1:]

                cu_seqlens = torch.cat([torch.zeros(1, device=seqlens.device, dtype=seqlens.dtype),
                                        seqlens.cumsum(0, dtype=torch.int32)], 0)

                q_block_table, kv_block_table, out_block_table, q_cache, kv_cache, out_cache, lse_cache = _blockize_qkv_out(
                    q, kv, cu_seqlens, cu_seqlens, block_size, block_size
                )

                # run benchmark with block table
                f = time_fwd(
                    flash_attn_varlen_kvpacked_func, q_cache, kv_cache, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, causal=causal, q_block_table=q_block_table, kv_block_table=kv_block_table, out_block_table=out_block_table, out_=out_cache, lse_=lse_cache, repeats=repeats
                )
                time_f[config, "Flash2-Paged-QKVO"] = f

                # run benchmark with kv table only
                f = time_fwd(
                    flash_attn_varlen_kvpacked_func, q, kv_cache, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, causal=causal, kv_block_table=kv_block_table, repeats=repeats
                )

                time_f[config, "Flash2-Paged-KV"] = f

                # run benchmark without block table
                f = time_fwd(
                    flash_attn_varlen_kvpacked_func, q, kv, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, causal=causal, repeats=repeats
                )

                time_f[config, "Flash2"] = f

                if attention_triton is not None:
                    q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                                        requires_grad=True) for _ in range(3)]
                    # Try both values of sequence_parallel and pick the faster one
                    try:
                        f = time_fwd(
                            attention_triton, q, k, v, causal, headdim**(-0.5),
                            False, repeats=repeats,
                        )
                    except Exception as e:
                        f = float('nan')
                    # try:
                    #     _, b0 = time_fwd(
                    #         attention_triton, q, k, v, causal, headdim**(-0.5),
                    #         True, repeats=repeats, verbose=False
                    #     )
                    # except:
                    #     b0 = float('inf')
                    time_f[config, "Triton"] = f
                    # time_b[config, "Triton"] = min(b, b0) if min(b, b0) < float('inf') else float('nan')

                if xops is not None:
                    q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                        requires_grad=True) for _ in range(3)]
                    f = time_fwd(
                        xops.memory_efficient_attention, q, k, v,
                        attn_bias=xops.LowerTriangularMask() if causal else None,
                        op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)
                    )
                    time_f[config, "xformers.c"] = f
                    # time_b[config, "xformers.c"] = b

                if xops is not None:
                    q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                        requires_grad=True) for _ in range(3)]
                    f = time_fwd(
                        xops.memory_efficient_attention, q, k, v,
                        attn_bias=xops.LowerTriangularMask() if causal else None,
                        op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
                    )
                    time_f[config, "xformers.f"] = f
                    # time_b[config, "xformers.f"] = b

                print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen}, block_size={block_size}###")
                for method in methods:
                    # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                        time_f[config, method]
                    )
                    # speed_b[config, method] = efficiency(
                    #     flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                    #     time_b[config, method]
                    # )
                    # speed_f_b[config, method] = efficiency(
                    #     flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                    #     time_f_b[config, method]
                    # )
                    print(
                        f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                        # f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                        # f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                    )


with open('flash2_attn_time.plk', 'wb') as fp:
    pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
