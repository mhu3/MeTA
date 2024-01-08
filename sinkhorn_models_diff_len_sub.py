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

from local_attention import LocalAttention
from product_key_memory import PKM
from reversible import ReversibleSequence, SequentialSequence, ModifiedSequentialSequence

# helper functions

def pad_sequences(sequences, padding_value=0.0):
    """
    Pads a list of sequences to the maximum length of the longest sequence.
    """
    max_len = max([seq.size(0) for seq in sequences])
    num_dim = sequences[0].size(1)

    padded_seqs = torch.full((len(sequences), max_len, num_dim), padding_value)
    
    for i, seq in enumerate(sequences):
        end = seq.size(0)
        padded_seqs[i, :end, :] = seq

    return padded_seqs

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

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

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


    return all_buckets

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

        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, max_seq_len, dim_heads * 2, non_permutative=non_permutative, temperature=temperature, sinkhorn_iter=sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, dim_heads, non_permutative=non_permutative, temperature=temperature, sinkhorn_iter=sinkhorn_iter, n_sortcut=n_sortcut)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def attention_calculation(self, b_q_i, b_k_j, b_v_j, R_ij):
        d_k = b_k_j.shape[-1]
        # Dot-product attention
        attention_score = torch.einsum('ie,je->ij', b_q_i, b_k_j) * (d_k ** -0.5)
        
        attention_weights = torch.softmax(attention_score, dim=-1)

        attention_matrix = torch.einsum('ij,je->ie', attention_weights, b_v_j)

        output = attention_matrix * R_ij

        return output

    def forward(self, q, k, v, bucket_size, q_mask=None, kv_mask=None):
        b, h, t, d_h = q.shape
        bh = b * h

        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        b_q = custom_bucket(q, bucket_size)
        b_k = custom_bucket(k, bucket_size)
        b_v = custom_bucket(v, bucket_size)

        R = self.sort_net(q, k, bucket_size)
        R = R.type_as(q).to(q.device)

        all_attention_matrices = []

        # Loop through each batch*head
        for idx in range(bh):
            # Slicing tensors for the specific batch and head
            b_q_sample = b_q[idx]
            b_k_sample = b_k[idx]
            b_v_sample = b_v[idx]

            summed_attention_matrices = []

            # Inside the double loop for 'i' and 'j'
            for i, b_q_i in enumerate(b_q_sample):

                attention_matrices_sample = []

                for j, b_k_j in enumerate(b_k_sample):

                    # Concatenate the local and the current b_k together
                    concatenated_b_k = torch.cat((b_k_sample[i], b_k_sample[j]), dim=0)
                    concatenated_b_v = torch.cat((b_v_sample[i], b_v_sample[j]), dim=0)

                    # # Skip the local attention part, it's already included as the first element
                    # if j == i:
                    #     continue
                    
                    R_ij = R[idx, i, j]

                    if R_ij <= 0.001:
                        # Append a zero tensor of appropriate shape to maintain consistent shape
                        attention_matrices_sample.append(torch.zeros_like(b_q_i))
                    else:
                        # Perform attention calculation
                        attention_matrices_sample.append(
                            self.attention_calculation(b_q_i, concatenated_b_k, concatenated_b_v, R_ij)
                            # self.attention_calculation(b_q_i, b_k_j, b_v_sample[j], R_ij)
                        )

                # Check that attention_matrices_sample is not empty before stacking
                if attention_matrices_sample:
                    global_attention_matrix = torch.stack(attention_matrices_sample).sum(dim=0)
                    combined_attention_matrix = global_attention_matrix
                    summed_attention_matrices.append(combined_attention_matrix)

            # Concatenate to represent that head
            all_attention_matrices.append(
                torch.cat(summed_attention_matrices, dim=0)
            )

        # Stack to get the final output
        outputs = torch.stack(all_attention_matrices)

        # Reshape
        outputs = outputs.reshape(b, h, t, d_h)

        return outputs

def apply_fn_after_split_ind(dim, ind, fn, t):
    l, r = split_at_index(dim, ind, t)
    return torch.cat((l, fn(r)), dim=dim)

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

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attention = LocalAttention(local_window_size, dropout = attn_dropout, look_forward=1)

        sink_heads = heads - n_local_attn_heads

        attn = SinkhornAttention(dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)

        self.sinkhorn_attention = attn

        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x, bucket_size, input_mask = None, context = None, context_mask = None):

        b, t, d, h, dh, l_h = *x.shape, self.heads, self.dim_head, self.n_local_attn_heads
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'
        assert not (context is not None and (context.shape[0], context.shape[2]) !=  (b, d)), 'contextual key / values must have the same batch and dimensions as the decoder'

        q = self.to_q(x)

        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)

        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))

        out = []

        if has_local > 0:
            out.append(self.local_attention(lq, lk, lv, input_mask = input_mask))

        if has_sinkhorn > 0:
            out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask, bucket_size=bucket_size))

        out = torch.cat(out, dim=1)
        out = split_heads(h, out)
        out = self.to_out(out)
        out = self.dropout(out)
        
        return out

class SinkhornSelfAttentionBlock(nn.Module):
    def __init__(self, dim, depth, max_seq_len = None, local_window_size = 8, heads = 8, dim_head = None, bucket_size = 64, kv_bucket_size = None, context_bucket_size = None, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., weight_tie = False, ff_glu = False, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 2, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 1,  pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        layers = nn.ModuleList([])

        kv_bucket_size = default(kv_bucket_size, bucket_size)
        context_bucket_size = default(context_bucket_size, bucket_size)

        get_attn = lambda: SinkhornSelfAttention(dim, max_seq_len, local_window_size = local_window_size, heads = heads, dim_head = dim_head, kv_bucket_size = kv_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, use_simple_sort_net = use_simple_sort_net, n_local_attn_heads = n_local_attn_heads, n_top_buckets = n_top_buckets)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, glu = ff_glu), along_dim=1)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

        get_attn_context = lambda: SinkhornSelfAttention(dim, max_seq_len, context_only = True, heads = heads, dim_head = dim_head, kv_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = context_n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, n_top_buckets = n_top_buckets)
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

        execute_type = ReversibleSequence if reversible else ModifiedSequentialSequence

        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn}

        self.layers = execute_type(layers, args_route = {**context_route_map, **attn_route_map}, layer_dropout = layer_dropout)
        self.receives_context = receives_context

        self.max_seq_len = max_seq_len
        self.context_bucket_size = context_bucket_size

    def forward(self, x, bucket_size):
        out = self.layers(x, bucket_size)
        # print("out shape: ", out.shape)
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

