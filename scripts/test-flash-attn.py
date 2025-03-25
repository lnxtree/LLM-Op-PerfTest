import torch
from models.model import FlashAtten
from utils.args import get_params, get_torch_dtype
import csv
import os

def test_flash_atten():
    args = get_params()
    if args is None:
        print("args is None")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name()}")  # 确认是否为 NVIDIA GPU
    flash_atten = FlashAtten(args)
    flash_atten.eval()
    input = torch.rand(args.seq_length, args.dim).to(dtype=get_torch_dtype(args.dtype), device=device)
    cu_seq_len = torch.arange(0, args.seq_length, step=512).to(dtype=torch.int32, device=device)
    max_seqlen = torch.max(cu_seq_len)
    output, elapsed_time_ms = flash_atten(input, cu_seq_len, max_seqlen)
    
    (qkv_time, flash_atten_time, linear_time) = elapsed_time_ms
    total_time = qkv_time + flash_atten_time + linear_time
    
    print(f"total_time: {total_time}ms")
    print(f"qkv_time: {qkv_time:.2f}ms")
    print(f"flash_atten_time: {flash_atten_time:.2f}ms")
    print(f"linear_time: {linear_time:.2f}ms")



def read_sample_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    inputs = [eval(line.strip()) for line in lines]
    return inputs

def test_flash_atten_input_sim_gamma():
    args = get_params()
    if args is None:
        print("args is None")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name()}")  # 确认是否为 NVIDIA GPU
    flash_atten = FlashAtten(args)
    flash_atten.eval()
    input = torch.rand(args.seq_length, args.dim).to(dtype=get_torch_dtype(args.dtype), device=device)
    
    
    sample_data = read_sample_from_file(f"./sample-out/sampled_lists_total-tokens-{args.seq_length}.txt")
    sample_data = sample_data[0:100]
    
    
    # warm up
    for seq_lens in sample_data[0:10]:
        seq_lens = torch.tensor(seq_lens)
        seq_cumsum = torch.cumsum(seq_lens, dim=-1, dtype=torch.int32)
        cu_seq_len = torch.cat((torch.tensor([0]), seq_cumsum), dim=-1).to(dtype=torch.int32, device=device)
        max_seqlen = torch.max(seq_lens)
        output, elapsed_time_ms = flash_atten(input, cu_seq_len, max_seqlen)
    
    if os.path.exists("./time-info") is False:
        os.mkdir("./time-info")
    file_path = f"./time-info/flash-attn-total-tokens-{args.seq_length}.csv"
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['total_tokes', 'max_seqlen', 'Atten_qkv_time', 'Atten_flash_time',  'Atten_linear_time',  'total_time'])

        for seq_lens in sample_data:
            seq_lens = torch.tensor(seq_lens)
            seq_cumsum = torch.cumsum(seq_lens, dim=-1, dtype=torch.int32)
            cu_seq_len = torch.cat((torch.tensor([0]), seq_cumsum), dim=-1).to(dtype=torch.int32, device=device)
            max_seqlen = torch.max(seq_lens)
            output, elapsed_time_ms = flash_atten(input, cu_seq_len, max_seqlen)
        
            (qkv_time, flash_atten_time, linear_time) = elapsed_time_ms
            total_time = qkv_time + flash_atten_time + linear_time
            
            # print(f"total_time: {total_time}ms")
            # print(f"qkv_time: {qkv_time:.2f}ms")
            # print(f"flash_atten_time: {flash_atten_time:.2f}ms")
            # print(f"linear_time: {linear_time:.2f}ms")
            
            writer.writerow([seq_cumsum[-1].item(), max_seqlen.item(), qkv_time, flash_atten_time, linear_time, total_time])
    
        
   
if __name__ == "__main__":
    test_flash_atten_input_sim_gamma()
