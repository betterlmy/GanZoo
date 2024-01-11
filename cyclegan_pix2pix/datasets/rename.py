import os

# 指定目录路径
directory = r'C:\Users\Eon\PycharmProjects\GanZoo\dataset\B301MM\low'

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.startswith('D') and filename.endswith('.png'):
        # 提取文件名中的数字部分
        number = filename[1:-4]

        # 构建新的文件名
        new_filename = number + '.png'

        # 构建完整的文件路径
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)

print("重命名完成")