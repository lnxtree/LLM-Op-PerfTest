import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

total_tokens = 16384

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

# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
scatter = ax.scatter(seqlen, batch_size, Atten_flash_time, 
                    c=Atten_flash_time,  # 用时间值来着色
                    cmap='viridis',      # 使用viridis颜色映射
                    marker='o')

# 添加标签和标题
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Batch Size')
ax.set_zlabel('Time (us)')
plt.title(f'Flash Attention Performance (total-tokens-{total_tokens})')

# 添加颜色条
plt.colorbar(scatter, label='Time (us)')

# 调整视角
ax.view_init(elev=20, azim=45)

# 保存和显示
plt.savefig(f'pyplot/3d_Flash-attn_seqlen_batch_time_total-tokens-{total_tokens}.png', 
            dpi=300, bbox_inches='tight')
plt.show()