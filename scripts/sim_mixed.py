import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
# 设置随机种子以保证可重复性
np.random.seed(42)

def generate_mixed_sequences(num_samples=30000, 
                           short_seq_mean=200,
                           long_seq_mean=2000,
                           short_seq_std=30,
                           long_seq_std=100,
                           long_seq_ratio=0.2):
    """
    生成混合序列长度分布
    
    参数:
    - num_samples: 总样本数
    - short_seq_mean: 短序列的平均长度
    - long_seq_mean: 长序列的平均长度
    - short_seq_std: 短序列的标准差
    - long_seq_std: 长序列的标准差
    - long_seq_ratio: 长序列占总样本的比例
    """
    
    # 计算长短序列的数量
    num_long = int(num_samples * long_seq_ratio)
    num_short = num_samples - num_long
    
    # 生成短序列
    short_sequences = np.random.normal(short_seq_mean, short_seq_std, num_short)
    
    # 生成长序列
    long_sequences = np.random.normal(long_seq_mean, long_seq_std, num_long)
    
    # 合并序列并取整
    all_sequences = np.concatenate([short_sequences, long_sequences])
    all_sequences = np.clip(np.round(all_sequences), 1, None)  # 确保序列长度至少为1
    np.random.shuffle(all_sequences)
    
    return all_sequences

# 生成序列
target_values = [200, 2000] 
sequences = generate_mixed_sequences(short_seq_mean=target_values[0],
                                      long_seq_mean=target_values[1])



# 进行 10 轮采样，使得总和 seq_len 约为 4096
num_samples = 30000
total_seqlens = [2048, 4096, 8192, 16384, 32768]


for total_seqlen in total_seqlens:
    sampled_seqlens = sequences
    sampled_seqlens = np.round(sampled_seqlens).astype(int)  # 取整

    queue = sampled_seqlens.tolist()
    sampled_lists = []
    current_list = []
    current_sum = 0
    while queue:
        value = queue.pop(0)
        if current_sum + value <= total_seqlen:
            current_list.append(value)
            current_sum += value
        else:
            sampled_lists.append(current_list)
            current_list = [value]
            current_sum = value
            
    for i, seq_lens in enumerate(sampled_lists):
        difference = total_seqlen - sum(seq_lens)
        if (len(seq_lens) == 0):
            seq_lens.append(difference)
            continue
        per_diference = difference // len(seq_lens)
        for j in range(len(seq_lens)):
            seq_lens[j] += per_diference
        seq_lens[0] += difference % len(seq_lens)

    work_dir = f"./sample-out/sampled_lists_distribution-{target_values[0]}-{target_values[1]}"
    if os.path.exists(work_dir) is False:
        os.makedirs(work_dir)

    with open(f"{work_dir}/total-tokens-{total_seqlen}.txt", "w") as f:
        for seq_lens in sampled_lists:
            f.write(str(seq_lens) + "\n")








# 绘制分布图
# plt.figure(figsize=(10, 6))

# # 绘制直方图
# plt.hist(sequences, bins=50, density=True, alpha=0.7, color='skyblue', label='序列分布')

# # 绘制理论分布曲线
# x = np.linspace(0, 3000, 1000)
# short_dist = norm.pdf(x, 100, 30) * 0.9  # 短序列分布
# long_dist = norm.pdf(x, 2000, 200) * 0.1  # 长序列分布
# mixed_dist = short_dist + long_dist

# plt.plot(x, mixed_dist, 'r-', lw=2, label='理论分布')

# plt.xlabel('序列长度')
# plt.ylabel('密度')
# plt.title('混合序列长度分布')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 设置x轴和y轴从0开始
# plt.xlim(left=0)
# plt.ylim(bottom=0)

# plt.savefig('mixed_sequence_distribution.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 打印一些统计信息
# print(f"生成的序列数量: {len(sequences)}")
# print(f"平均序列长度: {np.mean(sequences):.2f}")
# print(f"最短序列长度: {np.min(sequences):.0f}")
# print(f"最长序列长度: {np.max(sequences):.0f}")
# print(f"序列长度中位数: {np.median(sequences):.0f}")
