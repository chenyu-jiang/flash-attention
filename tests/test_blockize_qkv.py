import torch
import numpy as np

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
    out_cache = torch.zeros(q_cache_size, block_size_q, q_unpad.size(-2), q_unpad.size(-1), device=q_unpad.device, dtype=q_unpad.dtype)
    lse_cache = torch.zeros(q_unpad.size(-2), q_cache_size, block_size_q, device=q_unpad.device, dtype=torch.float)

    # pad block table to max_num_blocks_per_seq
    q_block_table = torch.stack([torch.cat([block, torch.zeros(max_num_blocks_per_seq_q - len(block), device=block.device, dtype=block.dtype)])
                     for block in q_block_table])
    kv_block_table = torch.stack([torch.cat([block, torch.zeros(max_num_blocks_per_seq_k - len(block), device=block.device, dtype=block.dtype)])
                        for block in kv_block_table])
    out_block_table = torch.stack([torch.cat([block, torch.zeros(max_num_blocks_per_seq_q - len(block), device=block.device, dtype=block.dtype)])
                        for block in out_block_table])
    return q_block_table, kv_block_table, out_block_table, q_cache, kv_cache, out_cache, lse_cache

def _reconstruct_blockized_out_lse(
    out_block_table: torch.Tensor,
    out_cache: torch.Tensor,
    lse_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
):
    # reverse cumsum to get the original seqlens
    q_seqlens_orig = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    block_size = out_cache.size(1)
    # out_unpad = [total_q, num_heads, head_size]
    out_unpad = torch.zeros(cu_seqlens_q[-1], out_cache.size(-2), out_cache.size(-1), device=out_cache.device, dtype=out_cache.dtype)
    # sm_lse = [num_heads, total_q,]
    sm_lse = torch.zeros(out_cache.size(-2), cu_seqlens_q[-1], device=lse_cache.device, dtype=torch.float)
    curr_q = 0
    for i, blocks in enumerate(out_block_table):
        curr_seq_q = 0
        for block_idx, block in enumerate(blocks):
            block = block.item()
            curr_block_seqlen = min(block_size, q_seqlens_orig[i].item() - block_idx * block_size)
            out_unpad[curr_q:curr_q + curr_block_seqlen] = out_cache[block, :curr_block_seqlen]
            sm_lse[:, curr_q:curr_q + curr_block_seqlen] = lse_cache[:, block, :curr_block_seqlen]
            curr_q += curr_block_seqlen
            curr_seq_q += curr_block_seqlen
            if curr_seq_q == q_seqlens_orig[i]:
                break
    return out_unpad, sm_lse

def test_blockize_qkv():
    q_seqlens = [4, 3, 5, 6, 9, 10]
    k_seqlens = [9, 3, 4, 5, 6, 8, 9]
    q_unpad = torch.randn(sum(q_seqlens), 2, 2)
    kv_unpad = torch.randn(sum(k_seqlens), 2, 2)
    cu_seqlens_q = torch.concat([torch.tensor([0], dtype=torch.int32), torch.tensor(q_seqlens, dtype=torch.int32).cumsum(0)], 0)
    cu_seqlens_k = torch.concat([torch.tensor([0], dtype=torch.int32), torch.tensor(k_seqlens, dtype=torch.int32).cumsum(0)], 0)
    block_size_q = 2
    block_size_k = 2
    q_block_table, kv_block_table, out_block_table, q_cache, kv_cache, out_cache, lse_cache = _blockize_qkv_out(q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, block_size_q, block_size_k)
    import code
    code.interact(local=locals())

def test_reconstruct_out_lse():
    n_head = 1
    block_size = 2
    head_dim = 4
    out_block_table = torch.tensor([
        [3, 4, 2, 0],
        [1, 5, 0, 0],
        [8, 9, 10, 11],
    ], dtype=torch.int32)
    out_cache = torch.randn(12, block_size, n_head, head_dim)
    lse_cache = torch.randn(n_head, 12, block_size)
    seqlens_q = torch.tensor([5, 4, 7], dtype=torch.int32)
    cu_seqlens_q = torch.concat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(0)], 0)
    out_unpad, sm_lse = _reconstruct_blockized_out_lse(out_block_table, out_cache, lse_cache, cu_seqlens_q)
    import code
    code.interact(local=locals())

# test_blockize_qkv()
test_reconstruct_out_lse()
