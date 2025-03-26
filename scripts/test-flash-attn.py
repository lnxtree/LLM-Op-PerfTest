import sys
sys.path.append('.')
import torch
from models.model import *
from utils.args import get_params, get_torch_dtype
def test_flash_atten():
    args = get_params()
    model_args = ModelArgs(
        world_size=args.world_size,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        seq_length=args.seq_length,
        dim=args.dim,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        dtype=args.dtype,
    )
    if args is None:
        print("args is None")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name()}")  # 确认是否为 NVIDIA GPU
    flash_atten = FlashAtten(args)
    flash_atten = flash_atten.to(dtype=get_torch_dtype(args.dtype), device=device)
    flash_atten.eval()
    input = torch.rand(args.seq_length, args.dim).to(dtype=get_torch_dtype(args.dtype), device=device)
    cu_seq_len = torch.arange(0, args.seq_length, step=512).to(dtype=torch.int32, device=device)
    max_seqlen = torch.max(cu_seq_len)
    
    for i in range(10):
        output, elapsed_time_ms = flash_atten(input, cu_seq_len, max_seqlen)
    
    output, elapsed_time_ms = flash_atten(input, cu_seq_len, max_seqlen)
    
    qkv_time, flash_atten_time, linear_time = elapsed_time_ms
    total_time = qkv_time + flash_atten_time + linear_time
    
    print(f"total_time: {total_time}us")
    print(f"qkv_time: {qkv_time:.2f}us")
    print(f"flash_atten_time: {flash_atten_time:.2f}us")
    print(f"linear_time: {linear_time:.2f}us")
    
    
if __name__ == "__main__":
    test_flash_atten()
