import argparse
import torch
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
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help='Hidden size.')
    parser.add_argument("--num_attention_heads", type=int, default=16,   
                        help='Number of attention heads.')
    parser.add_argument("--pipeline_model_parallel", type=int, default=1,
                        help='Degree of pipeline model parallelism.')
    parser.add_argument("--micro_batch", type=int, default=1,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.'
                        )
    parser.add_argument("--dtype", type=str, default="float16",
                        help='Data type.')
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=None,
        help="Transformer Feed-Forward Network hidden size. "
        "This is set to 4*hidden-size if not provided",
    )
    args = parser.parse_args()
    
    return args

def get_torch_dtype(dtype_str):
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.float32)