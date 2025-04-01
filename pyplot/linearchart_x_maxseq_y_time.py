import matplotlib.pyplot as plt
import csv
import os

total_tokens = [2048, 4096, 8192, 16384, 32768]

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
for total_token in total_tokens:
        
    file_path = f'./time-info/distribution-{target_min_seqlen}-{target_max_seqlen}/total-tokens-{total_token}-GQA-4.csv'
    seqlen = read_column_from_csv(file_path, 1)  
    batch_size = read_column_from_csv(file_path, 2)
    Atten_flash_time = read_column_from_csv(file_path, 4)


    time_ms = Atten_flash_time

    # 绘制折线图
    plt.figure(figsize=(8, 5))
    # plt.plot(seqlen, time_ms, marker='o', linestyle='-', color='b', label='Total time (ms)')

    # 添加散点
    plt.scatter(seqlen, time_ms, color='r')

    for x, y, batch_size in zip(seqlen, time_ms, batch_size):
        plt.text(x, y, f'({batch_size})', fontsize=5, ha='right')

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # 添加标签和标题
    plt.xlabel('Max Sequence Length')
    plt.ylabel('Total Time (us)')
    plt.title(f'Max Sequence Length vs Total Time on Flash-attn(total-tokens-{total_token})')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.savefig(f'{root_path}/linearchart_Flash-attn_tokens_vs_times_total-tokens-{total_token}.png')
    plt.show()