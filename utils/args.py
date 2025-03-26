import argparse
import torch
from typing import Optional

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1,
                        help='Degree of tensor model parallelism.')
    parser.add_argument("--seq_length", type=int, default=4096,
                        help='Sequence length.')
    parser.add_argument("--dim", type=int, default=2048,
                        help='Dimension.')
    parser.add_argument("--num_attention_heads", type=int, default=32,   
                        help='Number of attention heads.')
    parser.add_argument("--num_kv_heads", type=int, default=None,
                        help='Number of key-value heads. If None, use num_attention_heads.')
    parser.add_argument("--ffn_dim_multiplier", type=float, default=2.0,
                        help='Feed-forward network dimension multiplier. '
                        'ffn_dim = ffn_dim_multiplier * dim'
                        )
    parser.add_argument("--micro_batch", type=int, default=1,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.'
                        )
    parser.add_argument("--dtype", type=str, default="float16",
                        help='Data type.')
    args = parser.parse_args()
    
    return args

def get_torch_dtype(dtype_str):
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.float32)