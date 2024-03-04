import os
from tqdm import tqdm 
import shutil


def find_png_files(root_dir):
    png_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

# 替换 'your_directory_path' 为你的目标文件夹路径
# root_directory = r'E:\data\Preprocessed_256x256\256\Quarter Dose'
# png_files = find_png_files(root_directory)
# print(png_files)

# for index,path in tqdm(enumerate(png_files)):
#     file_name = os.path.basename(path)
#     shutil.copy(path, os.path.join(r"E:\data\AAPM_256\low",str(index)+".png"))  # 复制文件并保留元数据



import csv

# 打开CSV文件并指定编码格式（如果需要）
with open(r'E:\data\metadata.csv', 'r', encoding='utf-8') as file:
    # 创建CSV阅读器对象
    reader = csv.reader(file)
    # 跳过第一行（标题）
    next(reader)
    # 遍历每行数据
    for index,row in tqdm(enumerate(reader)):
        # print(row[4])
        # print(row[5])

        # print(row[4][3:])
        # print(os.path.join(r"E:\data\Preprocessed_256x256",r"256",row[4][4:]))
        shutil.copy(os.path.join(r"E:\data\Preprocessed_256x256",r"256",row[4][4:]), os.path.join(r"E:\data\AAPM_256\high",str(index)+".png"))  # 复制文件并保留元数据
        # break
        shutil.copy(os.path.join(r"E:\data\Preprocessed_256x256",r"256",row[5][4:]), os.path.join(r"E:\data\AAPM_256\low",str(index)+".png"))  # 复制文件并保留元数据
       

    