import matplotlib.pyplot as plt
import csv
import os

total_tokens_list = [2048, 4096, 8192, 16384, 32768]

def read_column_from_csv(file_path, column_index):
    column_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            if row:  # 确保行不为空
                column_data.append(float(row[column_index]))
    return column_data

target_min_seqlen = 200
target_max_seqlen = 2000

root_path = f"./pyplot/distribution-{target_min_seqlen}-{target_max_seqlen}"
if os.path.exists(root_path) is False:
    os.makedirs(root_path)
    
atten_average_times = []
for total_tokens in total_tokens_list:
    file_path = f'./time-info/distribution-{target_min_seqlen}-{target_max_seqlen}/total-tokens-{total_tokens}-GQA-4.csv'
    seqlen = read_column_from_csv(file_path, 1)  
    batch_size = read_column_from_csv(file_path, 2)
    Atten_flash_time = read_column_from_csv(file_path, 4)
    time_ms = Atten_flash_time
    atten_average_time = sum(time_ms) / len(time_ms)
    atten_average_times.append(atten_average_time)

# 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(total_tokens_list, atten_average_times, marker='o', linestyle='-', color='b', label='Flash-Attention Average Time (us)')

# 添加散点
plt.scatter(total_tokens_list, atten_average_times, color='r')

for x, y in zip(total_tokens_list, atten_average_times):
    plt.text(x, y, f'({x}, {y:.2f})', fontsize=9, ha='right')

# 添加标签和标题
plt.xlabel('Total Tokens')
plt.ylabel('Logit Average Time (us)')
plt.title('Total Tokens vs Flash-Attn Average Time')
plt.legend()
plt.grid(True)

# 显示图形
plt.savefig(f'{root_path}/total_tokens_vs_Flash-Attn_average_time.png')
plt.show()