import torch
import torch.cuda.memory as memory
from tqdm import tqdm
from einops import rearrange

# Set the fraction of GPU memory to be allocated
memory.set_per_process_memory_fraction(1.0)  # Use 80% of the available GPU memory
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.init as init

import math
from operator import mul
from math import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce

# from local_attention import LocalAttention
from product_key_memory import PKM
from reversible import ReversibleSequence, SequentialSequence, ModifiedSequentialSequence

# helper functions

def identity(x, *args, **kwargs): return x

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

def divisible_by(num, divisor):
    return num % divisor == 0

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def all_none(*arr):
    return all(el is None for el in arr)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def rotate_left(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(n, None))
    r = (*pre_slices, slice(0, n))
    return torch.cat((t[l], t[r]), dim=dim)

def rotate_right(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(-n, None))
    r = (*pre_slices, slice(None, -n))
    return torch.cat((t[l], t[r]), dim=dim)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def segment_sequence_based_on_speed(x, num_buckets=40, speed_threshold=30):
    bucket_lengths = []

    # Get the shape of the input tensor
    batch_size, num_rows, feature_dim = x.shape

    for batch in range(batch_size):  # Remove tqdm for simplicity
        current_bucket_length = 0
        current_bucket_lengths = []

        last_speed_greater_than_30 = None

        # Create buckets based on speed
        for i in range(num_rows):
            row = x[batch, i, :]
            speed = row[-1]  # Assuming speed is the last feature

            current_speed_condition = speed > speed_threshold

            if last_speed_greater_than_30 is not None and current_speed_condition != last_speed_greater_than_30:
                if current_bucket_length > 0:
                    current_bucket_lengths.append(current_bucket_length)
                current_bucket_length = 0

            current_bucket_length += 1
            last_speed_greater_than_30 = current_speed_condition

        # Finalize the last bucket if exists
        if current_bucket_length > 0:
            current_bucket_lengths.append(current_bucket_length)

        # Aggregation logic to adjust the number of buckets to num_buckets
        actual_num_buckets = len(current_bucket_lengths)

        while actual_num_buckets > num_buckets:
            min_len = min(current_bucket_lengths)
            min_len_idx = current_bucket_lengths.index(min_len)

            # Determine where to merge the smallest bucket
            if min_len_idx == 0:
                merge_idx = 1
            elif min_len_idx == len(current_bucket_lengths) - 1:
                merge_idx = len(current_bucket_lengths) - 2
            else:
                merge_idx = min_len_idx - 1 if current_bucket_lengths[min_len_idx - 1] < current_bucket_lengths[min_len_idx + 1] else min_len_idx + 1

            # Merge
            current_bucket_lengths[merge_idx] += current_bucket_lengths[min_len_idx]
            del current_bucket_lengths[min_len_idx]

            # Update actual number of buckets
            actual_num_buckets = len(current_bucket_lengths)

        bucket_lengths.append(current_bucket_lengths)

    return bucket_lengths

def custom_bucket(input_tensor, bucket_size):
    b_h, t, d_h = input_tensor.shape  # (batch_size * heads, seq_length, dim)
    num_samples = len(bucket_size)  # number of samples in the batch
    num_heads = b_h // num_samples  # number of heads

    all_buckets = []  # This will have a length of batch_size * num_heads

    for i in range(num_samples):
        for h in range(num_heads):
            head_sample_index = i * num_heads + h
            head_sample_tensor = input_tensor[head_sample_index, :, :]
            start = 0
            head_buckets = []
            for size in bucket_size[i]:
                end = start + size
                bucket = head_sample_tensor[start:end, :]
                head_buckets.append(bucket)
                start = end

            all_buckets.append(head_buckets)  # Append directly to all_buckets

    return all_buckets  # This will be a list of shape [batch_size * num_heads, num_buckets, bucket_size, dim]

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)

def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, bucket_size):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, bucket_size, **kwargs):
        x = self.norm(x)
        return self.fn(x, bucket_size, **kwargs)

class SimpleSortNet(nn.Module):
    def __init__(self, heads, max_buckets, dim, non_permutative, temperature, sinkhorn_iter):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.max_buckets = max_buckets
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets))
        self.act = nn.ReLU()

    def forward(self, q, k, bucket_size, topk=1):
        bh, = q.size(0)
        b = bh // self.heads

        # Using custom bucket function
        b_q_buckets = custom_bucket(q, bucket_size)
        b_k_buckets = custom_bucket(k, bucket_size)

        # Initialize lists to store the summed tensors for each sample in the batch
        b_q_sums_batch = []
        b_k_sums_batch = []

        # Loop through each sample in the batch
        for i in range(len(b_q_buckets)):
            # Sum along the sequence length for each bucket
            b_q_sums = torch.stack([bucket.sum(dim=0) for bucket in b_q_buckets[i]])
            b_k_sums = torch.stack([bucket.sum(dim=0) for bucket in b_k_buckets[i]])

            # Append the summed tensor for this sample to the list
            b_q_sums_batch.append(b_q_sums)
            b_k_sums_batch.append(b_k_sums)

        # Convert the list of tensors to a single tensor
        b_q_sums_batch = torch.stack(b_q_sums_batch)
        b_k_sums_batch = torch.stack(b_k_sums_batch)

        # Now, b_q_sums_batch and b_k_sums_batch should have a shape of [batch_size, num_buckets, feature_dim]
        sq = b_q_sums_batch
        sk = b_k_sums_batch
        
        x = torch.cat((sq, sk), dim=-1)

        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)

        return differentiable_topk(R, k=topk, temperature=self.temperature) if self.non_permutative else gumbel_sinkhorn(R, self.sinkhorn_iter, self.temperature)


class AttentionSortNet(nn.Module):
    def __init__(self, heads, dim, non_permutative, temperature, sinkhorn_iter, n_sortcut=0):
        super().__init__()
        self.bucket_size = None
        self.heads = heads
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

    def forward(self, q, k, bucket_size, topk=1):
        dim = self.dim
        b_h, t, d_h = q.shape  # b_h = batch_size * heads
        bucket_len = len(bucket_size[0])

        # Initialize a tensor to store all the R matrices for each combined head and batch
        all_R = torch.zeros(b_h, bucket_len, bucket_len, device=q.device)

        # Get the buckets using custom_bucket function
        b_q_buckets = custom_bucket(q, bucket_size)
        b_k_buckets = custom_bucket(k, bucket_size)

        # Loop through each combined head and batch
        for idx in range(b_h):
            # Sum along the sequence length for each bucket
            b_q_sums = torch.stack([bucket.sum(dim=0) for bucket in b_q_buckets[idx]])
            b_k_sums = torch.stack([bucket.sum(dim=0) for bucket in b_k_buckets[idx]])

            # Compute the attention score
            R = torch.einsum('ie,je->ij', b_q_sums, b_k_sums) * (dim ** -0.5)

            # Insert this R matrix into the correct position in all_R
            all_R[idx, :, :] = R
        
        # print("all_R.shape", all_R.shape)

        if self.non_permutative:
            k = topk if self.n_sortcut == 0 else self.n_sortcut
            return differentiable_topk(all_R, k=k)

        return gumbel_sinkhorn(F.relu(all_R), self.sinkhorn_iter, self.temperature)

class SinkhornAttention(nn.Module):
    def __init__(self, dim, dim_heads, heads, max_seq_len, temperature=0.75, non_permutative=False, sinkhorn_iter=7, n_sortcut=0, dropout=0., kv_bucket_size = None, use_simple_sort_net=False, n_top_buckets=1):
        super().__init__()

        self.dim = dim
        self.heads = heads  
        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut
        self.to_out = nn.Linear(2*dim_heads, dim_heads)

        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, max_seq_len, dim_heads * 2, non_permutative=non_permutative, temperature=temperature, sinkhorn_iter=sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, dim_heads, non_permutative=non_permutative, temperature=temperature, sinkhorn_iter=sinkhorn_iter, n_sortcut=n_sortcut)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bucket_size, q_mask=None, kv_mask=None):
        b, h, t, d_h = q.shape
        n_top = self.n_top_buckets
        bh = b * h

        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        b_q = custom_bucket(q, bucket_size)
        b_k = custom_bucket(k, bucket_size)
        b_v = custom_bucket(v, bucket_size)

        num_buckets = len(bucket_size[0])

        def pad_buckets_according_to_max_in_current_bucket(b_q, b_k, b_v, d_h):
            num_buckets = len(b_q[0])
            
            # 1. For each bucket index across all samples in the batch, compute the maximum length.
            max_lengths_per_bucket = [max(len(sample[i]) for sample in b_q) for i in range(num_buckets)]
            
            # Helper function to pad a list of tensors according to a max length
            def pad_bucket(bucket, max_len, d_h):
                # Get current lengths of all tensors in bucket
                lengths = [tensor.size(0) for tensor in bucket]
                # print(f"Original lengths in bucket: {lengths}")
                
                # Compute the number of zeros needed for each tensor to reach max_len
                padding_lengths = [max_len - length for length in lengths]
                
                # Create the padded bucket
                padded_bucket = [torch.cat([tensor, torch.zeros(pad_len, d_h, device=tensor.device)], dim=0) for tensor, pad_len in zip(bucket, padding_lengths)]
                
                # Stack the tensors in the padded bucket along the 0th dimension
                stacked_bucket = torch.stack(padded_bucket, dim=0)
                
                return stacked_bucket, lengths

            # For each bucket, pad tensors to max length of that bucket
            b_q_padded, lengths_q = zip(*[pad_bucket([sample[i] for sample in b_q], max_lengths_per_bucket[i], d_h) for i in range(num_buckets)])
            # print(f"Original lengths for q: {lengths_q}")
            b_k_padded, lengths_k = zip(*[pad_bucket([sample[i] for sample in b_k], max_lengths_per_bucket[i], d_h) for i in range(num_buckets)])
            b_v_padded, lengths_v = zip(*[pad_bucket([sample[i] for sample in b_v], max_lengths_per_bucket[i], d_h) for i in range(num_buckets)])
            
            # Compute the masks once the tensors are padded
            masks_q = [[torch.cat([torch.ones(length, device=tensor.device), torch.zeros(max_len - length, device=tensor.device)], dim=0) for tensor, length, max_len in zip(bucket, bucket_lengths, [max_lengths_per_bucket[i]]*len(bucket))] for i, (bucket, bucket_lengths) in enumerate(zip(b_q_padded, lengths_q))]
            masks_k = [[torch.cat([torch.ones(length, device=tensor.device), torch.zeros(max_len - length, device=tensor.device)], dim=0) for tensor, length, max_len in zip(bucket, bucket_lengths, [max_lengths_per_bucket[i]]*len(bucket))] for i, (bucket, bucket_lengths) in enumerate(zip(b_k_padded, lengths_k))]
            masks_v = [[torch.cat([torch.ones(length, device=tensor.device), torch.zeros(max_len - length, device=tensor.device)], dim=0) for tensor, length, max_len in zip(bucket, bucket_lengths, [max_lengths_per_bucket[i]]*len(bucket))] for i, (bucket, bucket_lengths) in enumerate(zip(b_v_padded, lengths_v))]
            
            return b_q_padded, masks_q, b_k_padded, masks_k, b_v_padded, masks_v, lengths_q

        b_q_pad_list, masks_q_list, b_k_pad_list, masks_k_list, b_v_pad_list, masks_v_list, lengths_q_list = pad_buckets_according_to_max_in_current_bucket(b_q, b_k, b_v, d_h)
        
        # print("b_q_pad_list.shape", len(b_q_pad_list))
        # print("b_q_pad_list[0].shape", len(b_q_pad_list[0]))
        # print("b_q_pad_list[0][0].shape", b_q_pad_list[0][0].shape)

        masks_q_tensor_list = [torch.stack(masks_q, dim=0) for masks_q in masks_q_list]
        masks_k_tensor_list = [torch.stack(masks_k, dim=0) for masks_k in masks_k_list]

        R = self.sort_net(q, k, bucket_size)
        # print("R shape: ", R.shape)
        R = R.type_as(q).to(q)

        all_attention_matrices = []
        all_local_attention_matrices = []

        # Loop through each bucket in b_q
        for i, b_q_i in enumerate(b_q_pad_list):
            R_values = R[:, i, :]
            concatenated_attention_matrices = []
            # Local attention: attention within the same bucket
            mask_q_i = masks_q_tensor_list[i]
            mask_k_j = masks_k_tensor_list[i]  # For local attention, k and q are from the same bucket
            mask_combined = mask_q_i.unsqueeze(-1) * mask_k_j.unsqueeze(-2)
            local_attention_score = torch.einsum('bie,bje->bij', b_q_i, b_k_pad_list[i]) * (d_h ** -0.5)
            local_attention_score = local_attention_score.masked_fill(mask_combined == 0, -1e9)
            local_attention_weights = torch.softmax(local_attention_score, dim=-1)
            local_attention_matrix = torch.einsum('bij,bje->bie', local_attention_weights, b_v_pad_list[i])
            local_masked_output = local_attention_matrix * mask_q_i.unsqueeze(-1).type_as(local_attention_matrix)
            all_local_attention_matrices.append(local_masked_output)

            # Loop through each bucket in b_k and b_v
            for j, b_k_j in enumerate(b_k_pad_list):
                mask_k_j = masks_k_tensor_list[j]
                # Check if the corresponding R value is 0, first dimension is batch size, second dimension is number of buckets, third dimension is number of buckets
                # if R_values[:, j].sum() == 0:
                if R_values[:, j].sum() < 1:
                    continue
                else:
                    
                    mask_combined = mask_q_i.unsqueeze(-1) * mask_k_j.unsqueeze(-2)               
                    # Calculate the attention matrix
                    attention_score = torch.einsum('bie,bje->bij', b_q_i, b_k_j) * (d_h ** -0.5)
                    # print("attention weights shape: ", attention_weights.shape)
                    attention_score = attention_score.masked_fill(mask_combined == 0, -1e9)
                    attention_weights = torch.softmax(attention_score, dim=-1)
                    
                    # Calculate attention matrix
                    attention_matrix = torch.einsum('bij,bje->bie', attention_weights, b_v_pad_list[j])
                    # print("attention matrix shape: ", attention_matrix.shape)

                    attention_matrix *= R_values[:, j][:, None, None] # Element-wise multiply by R[i, j]
                    # print("attention matrix shape after R multiplication: ", attention_matrix.shape)
                    concatenated_attention_matrices.append(attention_matrix)

            # Convert the list of attention matrices to a single tensor
            concatenated_attention_matrix_tensor = torch.stack(concatenated_attention_matrices, dim=1)
            final_attention_matrix = torch.sum(concatenated_attention_matrix_tensor, dim=1)
            masked_output = final_attention_matrix * masks_q_tensor_list[i].unsqueeze(-1).type_as(final_attention_matrix)
            all_attention_matrices.append(masked_output)

        # combined_matrices = []

        # for all_attention, local_attention, unpadded_len in zip(all_attention_matrices, all_local_attention_matrices, lengths_q_list):
        #     output = all_attention + local_attention
        #     # output = torch.cat((all_attention, local_attention), dim=1)
    
        #     actual_len = unpadded_len[0]
        #     print("actual_len", actual_len)
        #     print("output.shape", output.shape[1])
            
        #     # If the output exceeds the unpadded length, trim it
        #     if output.shape[1] > actual_len:
        #         combined_matrices.append(output[:, :actual_len, :])
        #     else:
        #         combined_matrices.append(output)
        # print("combined_matrices.shape", len(combined_matrices))
        # print("combined_matrices[0].shape", combined_matrices[0].shape)
        # print("combined_matrices[0][0].shape", combined_matrices[0][0].shape)

        all_m = []
        local_m = []

        for all_attention, local_attention, unpadded_len in zip(all_attention_matrices, all_local_attention_matrices, lengths_q_list):
            actual_len = unpadded_len[0]
            
            if all_attention.shape[1] > actual_len:
                all_m.append(all_attention[:, :actual_len, :])
            else:
                all_m.append(all_attention)
                
            if local_attention.shape[1] > actual_len:
                local_m.append(local_attention[:, :actual_len, :])
            else:
                local_m.append(local_attention)

        all_output = torch.cat(all_m, dim=1)
        local_output = torch.cat(local_m, dim=1)

        all_output = all_output.reshape(b, h, t, d_h)
        local_output = local_output.reshape(b, h, t, d_h)

        # # outputs = torch.cat((all_output, local_output), dim=3)

        # # outputs = self.to_out(outputs)
        # # Use the learnable weight to combine all_output and local_output
        # weight = torch.nn.Parameter(torch.randn(1)).to(all_output.device)  # Ensure weight is on the same device as all_output
        # outputs = weight * all_output + (1 - weight) * local_output

        # # # outputs = torch.cat(combined_matrices, dim=1)
        # # # outputs = outputs[:, :t, :] 
        # # print("outputs.shape", outputs.shape)
        # # outputs = outputs.reshape(b, h, t, d_h)
        # # print("outputs.shape", outputs.shape)

        return all_output, local_output


class SinkhornSelfAttention(nn.Module):
    def __init__(self, dim, max_seq_len, local_window_size=8, heads = 8, dim_head = None, kv_bucket_size = None, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, attn_dropout = 0., dropout = 0., context_only = False, use_simple_sort_net = False, n_local_attn_heads = 0, n_top_buckets = 1):
        super().__init__()
        assert dim_head or divisible_by(dim, heads), f'If dim_head is None, dimension {dim} must be divisible by the number of heads {heads}'
        assert n_local_attn_heads <= heads, 'number of local attention heads cannot exceed total heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        self.heads = heads
        self.context_only = context_only
        self.to_q = nn.Linear(dim, dim_heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_heads * 2, bias=False) if not context_only else None

        self.to_out = nn.Linear(dim_heads, dim)

        # self.n_local_attn_heads = n_local_attn_heads
        # self.local_attention = LocalAttention(local_window_size, dropout = attn_dropout, look_forward=1)

        sink_heads = heads - n_local_attn_heads

        attn = SinkhornAttention(dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)
        self.sinkhorn_attention = attn
        
        # global_atten, local_atten = SinkhornAttention(dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)
        # self.sinkhorn_attention = global_atten
        # self.local_attention = local_atten
        sinkhorn_attention_instance = SinkhornAttention(dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)
        self.attention = sinkhorn_attention_instance
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x, bucket_size, input_mask = None, context = None, context_mask = None):

        # b, t, d, h, dh, l_h = *x.shape, self.heads, self.dim_head, self.n_local_attn_heads
        b, t, d, h, dh, l_h = *x.shape, self.heads*2, self.dim_head, self.heads
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'
        assert not (context is not None and (context.shape[0], context.shape[2]) !=  (b, d)), 'contextual key / values must have the same batch and dimensions as the decoder'

        q = self.to_q(x)

        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)
        # print("q.shape", q.shape, "k.shape", k.shape, "v.shape", v.shape)
 
        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        # print("lq.shape", lq.shape)
        # print("q.shape", q.shape)

        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))
        # print("has_local", has_local)
        # print("has_sinkhorn", has_sinkhorn)

        global_attention, local_attention = self.attention(q, k, v, bucket_size, q_mask = input_mask, kv_mask = kv_mask)
        out = []
        out.append(global_attention)
        # print("out.shape", len(out))
        # print("out[0].shape", out[0].shape)
        out.append(local_attention)
        # print("out.shape", len(out))
        # print("out[0].shape", out[0].shape)

        
        # if has_local > 0:
        #     out.append(self.local_attention(lq, lk, lv, input_mask = input_mask))

        # if has_sinkhorn > 0:
        #     out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask, bucket_size=bucket_size))
        # print("out.shape1", len(out))
        # print("out[0].shape", out[0].shape)
        out = torch.cat(out, dim=1)
        out = split_heads(h, out)
        out = self.to_out(out)
        out = self.dropout(out)
        
        return out
    
class SinkhornSelfAttentionBlock(nn.Module):
    def __init__(self, dim, depth, max_seq_len = None, local_window_size = 8, heads = 8, dim_head = None, bucket_size = 64, kv_bucket_size = None, context_bucket_size = None, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., weight_tie = False, ff_glu = False, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 2, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 1,  pkm_layers = tuple(), pkm_num_keys = 128):
        super(SinkhornSelfAttentionBlock, self).__init__()
    #     self.embed_dim = dim
    #     self.num_heads = heads

    #     self.attention = SinkhornSelfAttention(dim, max_seq_len, local_window_size = local_window_size, heads = heads, dim_head = dim_head, kv_bucket_size = kv_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, use_simple_sort_net = use_simple_sort_net, n_local_attn_heads = n_local_attn_heads, n_top_buckets = n_top_buckets)
    #     self.dropout = nn.Dropout(attn_dropout)
    #     self.layer_norm = nn.LayerNorm(dim)

    #     self.feed_forward = nn.Sequential(
    #         nn.Linear(dim, 4 * dim),
    #         nn.ReLU(),
    #         nn.Linear(4 * dim,dim),
    #         nn.Dropout(attn_dropout)
    #     )

    # def forward(self, x, bucket_size, mask = None):
    #     batch_size, seq_len, embed_dim = x.size()
    #     attention_output = self.attention(x, bucket_size, input_mask = mask)  
    #     x = self.layer_norm(x + self.dropout(attention_output))
    #     ff_output = self.feed_forward(x)
    #     x = self.layer_norm(x + self.dropout(ff_output))
    #     return x
        layers = nn.ModuleList([])

        kv_bucket_size = default(kv_bucket_size, bucket_size)
        context_bucket_size = default(context_bucket_size, bucket_size)

        get_attn = lambda: SinkhornSelfAttention(dim, bucket_size, max_seq_len,  heads = heads, dim_head = dim_head, kv_bucket_size = kv_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, use_simple_sort_net = use_simple_sort_net, n_local_attn_heads = n_local_attn_heads, n_top_buckets = n_top_buckets)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, glu = ff_glu), along_dim=1)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

        get_attn_context = lambda: SinkhornSelfAttention(dim, bucket_size, max_seq_len, context_only = True, heads = heads, dim_head = dim_head, kv_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = context_n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, n_top_buckets = n_top_buckets)
        get_ff_context = lambda: FeedForward(dim, dropout = ff_dropout, glu = ff_glu)

        if weight_tie:
            get_attn, get_attn_context, get_ff, get_ff_context = map(cache_fn, (get_attn, get_attn_context, get_ff, get_ff_context))

        fn_wrapper = partial(PreNorm, nn.LayerNorm, dim) if not use_rezero else ReZero

        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in pkm_layers

            get_parallel_fn = get_ff if not use_pkm else get_pkm

            layers.append(nn.ModuleList([
                fn_wrapper(get_attn()),
                fn_wrapper(get_parallel_fn())
            ]))

            if not receives_context:
                continue

            layers.append(nn.ModuleList([
                fn_wrapper(get_attn_context()),
                fn_wrapper(get_ff_context())
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn}

        self.layers = execute_type(layers, args_route = {**context_route_map, **attn_route_map}, layer_dropout = layer_dropout)
        self.receives_context = receives_context

        self.max_seq_len = max_seq_len
        self.pad_to_bucket_size = lcm(bucket_size, kv_bucket_size)
        self.context_bucket_size = context_bucket_size

        self.is_fixed_length = use_simple_sort_net
        # if not using attention sort and also not causal, force fixed sequence length
        assert not (self.is_fixed_length and self.max_seq_len is None), 'maximum sequence length must be specified if length is fixed'

    def forward(self, x, bucket_size, **kwargs):
        assert not (self.is_fixed_length and x.shape[1] != self.max_seq_len), f'you must supply a sequence of length {self.max_seq_len}'
        assert ('context' not in kwargs or self.receives_context), 'needs to be initted with receives_context True if passing contextual key / values'
        out = self.layers(x, bucket_size, **kwargs)
        return out

class SinkhornTransformer(nn.Module):
    def __init__(self, sub_len, num_tokens, dim, depth, local_window_size = 8, heads = 8, dim_head = None, num_buckets = 40, kv_bucket_size = None, context_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_glu = False, return_embeddings = False, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., emb_dropout = 0., weight_tie = False, emb_dim = None, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 0, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 2, pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.num_buckets = num_buckets

        # number of sub_trajectories
        self.sub_len = sub_len
        self.to_token_emb = nn.Linear(num_tokens, emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.fc = nn.Sequential(
            nn.Linear(sub_len * emb_dim, 64),  # Adjust the input dimension here
            nn.Dropout(0.1),
            nn.ReLU(),
        )    

        self.sim = nn.Sequential(
            nn.Linear(128, 1),
            nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

        for layer in self.sim:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

        self.sinkhorn_transformer = SinkhornSelfAttentionBlock(dim, depth, local_window_size = local_window_size, heads = heads, dim_head = dim_head, kv_bucket_size = kv_bucket_size, context_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, reversible = reversible, ff_chunks = ff_chunks, ff_dropout = ff_dropout, attn_dropout = attn_dropout, attn_layer_dropout = attn_layer_dropout, layer_dropout = layer_dropout, weight_tie = weight_tie, ff_glu = ff_glu, use_simple_sort_net = use_simple_sort_net, receives_context = receives_context, context_n_sortcut = context_n_sortcut, n_local_attn_heads = n_local_attn_heads, use_rezero = use_rezero, n_top_buckets = n_top_buckets,  pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys)
    
    def forward(self, x):
        
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]

        bucket_size_1 = segment_sequence_based_on_speed(x1, num_buckets=self.num_buckets)
        bucket_size_2 = segment_sequence_based_on_speed(x2, num_buckets=self.num_buckets)

        x1 = self.to_token_emb(x1)
        x2 = self.to_token_emb(x2)     
            
        x1 = self.sinkhorn_transformer(x1, bucket_size=bucket_size_1)
        x2 = self.sinkhorn_transformer(x2, bucket_size=bucket_size_2)

        outputs = []
        x1_list = []
        x2_list = []
        for i in range(x1.shape[1] // self.sub_len):
            # x1.shape[1] // self.num_sub_len is the length of the sub_trajectory
            sub_x1 = x1[:, i *self.sub_len: (i + 1) * self.sub_len, :]
            sub_x2 = x2[:, i *self.sub_len: (i + 1) * self.sub_len, :]

            embedding_x1 = torch.flatten(sub_x1, start_dim=1)
            embedding_x2 = torch.flatten(sub_x2, start_dim=1)

            x1_list.append(embedding_x1)
            x2_list.append(embedding_x2)

        for i in range(len(x1_list)):
            for j in range(i+1, len(x1_list)):
                outputs.append(self.sim((torch.cat((self.fc(x1_list[i]), self.fc(x1_list[j])), dim=1))))
            for k in range(len(x2_list)):
                outputs.append(self.sim((torch.cat((self.fc(x1_list[i]), self.fc(x2_list[k])), dim=1))))
        for i in range(len(x2_list)):
            for j in range(i+1, len(x2_list)):
                outputs.append(self.sim((torch.cat((self.fc(x2_list[i]), self.fc(x2_list[j])), dim=1))))
       
        outputs = torch.cat(outputs, dim=1)
        # print("outputs.shape", outputs.shape)
        return outputs

class SinkhornViTBinaryClassifier(nn.Module):
    def __init__(self, sub_len, num_tokens, dim, depth, local_window_size = 0, heads=8, dim_head=None, num_buckets = 40, non_permutative=False, sinkhorn_iter=5, n_sortcut=0, temperature=0.75, reversible=False, ff_chunks=1, ff_glu=False, return_embeddings=False, ff_dropout=0., attn_dropout=0., attn_layer_dropout=0., layer_dropout=0., emb_dropout=0., weight_tie=False, emb_dim=None, use_simple_sort_net=None, receives_context=False, context_n_sortcut=0, n_local_attn_heads=0, use_rezero=False, n_top_buckets=1, pkm_layers=tuple(), pkm_num_keys=128):
        super(SinkhornViTBinaryClassifier, self).__init__()
        self.model = SinkhornTransformer(
            sub_len=sub_len,
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            local_window_size=local_window_size,
            heads=heads,
            dim_head=dim_head,
            num_buckets=num_buckets,
            non_permutative=non_permutative,
            sinkhorn_iter=sinkhorn_iter,
            n_sortcut=n_sortcut,
            temperature=temperature,
            return_embeddings=return_embeddings,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            attn_layer_dropout=attn_layer_dropout,
            layer_dropout=layer_dropout,
            emb_dropout=emb_dropout,
            weight_tie=weight_tie,
            emb_dim=emb_dim,
            use_simple_sort_net=use_simple_sort_net,
            receives_context=receives_context,
            context_n_sortcut=context_n_sortcut,
            n_local_attn_heads=n_local_attn_heads,
        )

    def forward(self, x):
        prediction = self.model(x)
        return prediction
