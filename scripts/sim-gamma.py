import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
# 目标区间
x_min, x_max = 0, 4096

# 目标分布占比
percentiles = [0.2, 0.8]  # 20% 和 80% 分位点
target_values = [600, 1000]  # 对应 350 和 600 tokens

# 定义误差函数，用于优化 gamma 分布参数
def loss(params):
    alpha, beta = params
    gamma_dist = stats.gamma(alpha, scale=1/beta)
    errors = [(gamma_dist.ppf(p) - v) ** 2 for p, v in zip(percentiles, target_values)]
    return sum(errors)

# 初始参数猜测值
initial_guess = [5, 0.005]

# 进行优化，确保参数合理
bounds = [(0.1, 100), (0.0001, 1)]
result = minimize(loss, initial_guess, bounds=bounds, method="L-BFGS-B")
alpha_opt, beta_opt = result.x

# 生成优化后的 gamma 分布
x = np.linspace(x_min, x_max, 1000)
gamma_dist_opt = stats.gamma.pdf(x, a=alpha_opt, scale=1/beta_opt)

# 归一化使其积分为 1
gamma_dist_opt /= np.trapz(gamma_dist_opt, x)


# 进行 10 轮采样，使得总和 seq_len 约为 4096
num_samples = 4000
total_seqlen = 32768
sampled_seqlens = stats.gamma.rvs(alpha_opt, scale=1/beta_opt, size=num_samples)
sampled_seqlens = np.round(sampled_seqlens).astype(int)  # 取整

queue = sampled_seqlens.tolist()
# 从队列中取数，每次取接近 total_seqlen 的数为一个列表
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
    per_diference = difference // len(seq_lens)
    for j in range(len(seq_lens)):
        seq_lens[j] += per_diference
    seq_lens[0] += difference % len(seq_lens)


if os.path.exists("sample-out") is False:
    os.makedirs("sample-out")

with open(f"sample-out/sampled_lists_total-tokens-{total_seqlen}.txt", "w") as f:
    for seq_lens in sampled_lists:
        f.write(str(seq_lens) + "\n")
# print(sampled_lists)

# 画出优化后的分布曲线
plt.figure(figsize=(8, 5))
plt.plot(x, gamma_dist_opt, label=f"Optimized Gamma (α={alpha_opt:.2f}, β={beta_opt:.5f})", color='r')
plt.axvline(target_values[0], linestyle="--", color="gray", label="Target {}".format(target_values[0]))
plt.axvline(target_values[1], linestyle="--", color="gray", label="Target {}".format(target_values[1]))
plt.xlabel("Sequence Length")
plt.ylabel("Probability Density")
plt.title("Optimized Gamma Distribution for Sequence Length")
plt.legend()
plt.grid()
plt.show()

# 输出优化后的参数
alpha_opt, beta_opt
