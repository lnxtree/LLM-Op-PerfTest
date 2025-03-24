
import torch
import time
import warnings
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.utils import divide, cuda_timing_decorator
try:
    from einops import rearrange
except ImportError as e:
    rearrange = None
    print("Failed to import 'einops'. Functions using 'rearrange' might not work.")
from typing import Callable, Optional

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
        


class FlashAtten(nn.Module):
    def __init__(self, args=None):
        super(FlashAtten, self).__init__()
        self.tp = args.tensor_model_parallel_size
        micro_batch = args.micro_batch
        seq_len = args.seq_length

        dim = args.dim
        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()

        self.atten_weight_1 = torch.rand(
            divide((3 * hidden_size), self.tp), dim, device=device
        ).to(dtype)

        self.hidden_size_per_partition = divide(hidden_size, self.tp)
        self.num_attention_heads_per_partition = divide(num_attention_heads, self.tp)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        self.atten_linear_weight = torch.rand(
            dim, self.hidden_size_per_partition, device=device
        ).to(dtype)

    @cuda_timing_decorator
    def _apply_attenqkv(self, input):

        output = F.linear(input , self.atten_weight_1)


        (query_layer, key_layer, value_layer) = torch.split(
            output,
            [
                self.hidden_size_per_partition,
                self.hidden_size_per_partition,
                self.hidden_size_per_partition,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            -1,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        key_layer = key_layer.view(
            -1,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        value_layer = value_layer.view(
            -1,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        return query_layer, key_layer, value_layer

    @cuda_timing_decorator
    def _apply_flash_atten(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q = cu_seqlens_q,
            cu_seqlens_k = cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=None,
            causal=True,
        )

        return output

    @cuda_timing_decorator
    def _apply_Linear(self, context_layer):
        context_layer = rearrange(context_layer, "t h d -> t (h d)").contiguous()
        output_parallel = F.linear(context_layer, self.atten_linear_weight)
        return output_parallel

    def forward(self, input, cu_seqlens, max_seqlen):
        # input: [batch seq_len hidden_size]
        # q, k, v: [total_seq_len hidden_size]
        # context_layer: [total_seq_len hidden_size]
        result, qkv_time = self._apply_attenqkv(input)
        q, k, v = result

        context_layer, flash_time = self._apply_flash_atten(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        output, attrn_linear_time = self._apply_Linear(context_layer)
        return output, qkv_time, flash_time, attrn_linear_time

