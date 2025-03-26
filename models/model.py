
import torch
import time
import warnings
from dataclasses import dataclass
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
        
@dataclass
class ModelArgs:
    world_size: int = 1
    tensor_model_parallel_size: int = 1
    seq_length: int = 4096
    dim: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: Optional[int] = None
    vocab_size: int = -1
    ffn_dim_multiplier: Optional[float] = None
    dtype: str = "float16"

    

class FlashAtten(nn.Module):
    def __init__(self, args: ModelArgs):
        super(FlashAtten, self).__init__()
        self.tp = args.tensor_model_parallel_size
        seq_len = args.seq_length

        self.dim = args.dim
        self.head_dim = self.dim // args.num_attention_heads
        self.num_attention_heads = args.num_attention_heads
        self.num_kv_heads = args.num_kv_heads
        
        if self.num_kv_heads is None:
            self.num_kv_heads = args.num_attention_heads
        self.ffn_dim = int(args.ffn_dim_multiplier * self.dim)
        
        num_attention_heads = args.num_attention_heads
        num_kv_heads = self.num_kv_heads
        
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()

        self.wq = nn.Linear(self.dim, num_attention_heads * self.head_dim)
        self.wk = nn.Linear(self.dim, num_kv_heads * self.head_dim)
        self.wv = nn.Linear(self.dim, num_kv_heads * self.head_dim)

        self.wo = nn.Linear(num_attention_heads * self.head_dim, self.dim)

    @cuda_timing_decorator
    def _apply_attenqkv(self, input):

        q, k, v = self.wq(input), self.wk(input), self.wv(input)
        q = q.view(-1, self.num_attention_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        return q, k, v

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
    def _apply_Linear(self, input):
        input = input.view(-1, self.dim)
        output = self.wo(input)
        return output

    def forward(self, input, cu_seqlens, max_seqlen):
        """
            input: [total_tokens, hidden_size]
            q: [total_seq_len num_attention_heads head_dim]
            k,v : [total_seq_len num_kv_heads head_dim]
            context_layer: [total_seq_len num_attention_heads head_dim]
            output: [total_seq_len hidden_size]
        """
        
        result, qkv_time = self._apply_attenqkv(input)
        q, k, v = result

        context_layer, flash_time = self._apply_flash_atten(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        output, attrn_linear_time = self._apply_Linear(context_layer)
        return output, (qkv_time, flash_time, attrn_linear_time)

