import matplotlib.pyplot as plt
import csv


total_tokens = 32768

def read_column_from_csv(file_path, column_index):
    column_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            if row:  # 确保行不为空
                column_data.append(float(row[column_index]))
    return column_data

file_path = f'./time-info/flash-attn-total-tokens-{total_tokens}.csv'
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



# 添加标签和标题
plt.xlabel('Sequence Length')
plt.ylabel('Total Time (us)')
plt.title(f'Sequence Length vs Total Time on Flash-attn(total-tokens-{total_tokens})')
plt.legend()
plt.grid(True)

# 显示图形
plt.savefig(f'pyplot/linearchart_Flash-attn_tokens_vs_times_total-tokens-{total_tokens}.png')
plt.show()