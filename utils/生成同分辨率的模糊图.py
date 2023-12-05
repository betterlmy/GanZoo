import os
import shutil

from PIL import Image
from torchvision.transforms import functional as TF


def resize_images(input_folder, original_size=(256, 256), low_res_size=(64, 64), gray=True):
    path = os.path.dirname(input_folder)
    output_folder = os.path.join(path, 'low_gray' if gray else 'low')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    if gray:
        gray_folder = os.path.join(path, 'high_gray')
        # 先保存原始图像转为灰度图的图像
        if not os.path.exists(gray_folder):
            os.makedirs(gray_folder)

        else:
            shutil.rmtree(output_folder)
            os.makedirs(gray_folder)

    for img_name in os.listdir(input_folder):
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path)
        if gray:
            if img.mode != 'L':
                img = img.convert('L')
                img.save(os.path.join(gray_folder, img_name))
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        # 确保图像是原始大小
        img = TF.resize(img, original_size)

        # 缩小图像
        img_low_res = TF.resize(img, low_res_size)

        # 放大图像
        img_low_res_upsampled = TF.resize(img_low_res, original_size, interpolation=Image.BICUBIC)

        # 保存图像
        img_low_res_upsampled.save(os.path.join(output_folder, img_name))


if __name__ == '__main__':
    # 使用函数
    input_folder = '/Users/zane/PycharmProjects/GanZoo/dataset/celeba/high'  # 原始图像文件夹路径

    resize_images(input_folder)
