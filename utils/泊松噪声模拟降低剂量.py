import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def add_poisson_noise(image, scale=1.0):
    """
    Apply Poisson noise to the input image.
    :param image: input image
    :param scale: scale factor for the input image
    :return: noisy image
    """
    # Scale the input image
    scaled_image = image * scale

    # Apply Poisson noise
    noisy_image = np.random.poisson(scaled_image).astype(np.uint8)

    return noisy_image


if __name__ == '__main__':

    # 创建文件夹
    path = r"C:\Users\Eon\PycharmProjects\GanZoo\dataset\B301MM"
    pathA = os.path.join(path, "poisson_noisyA")
    pathB = os.path.join(path, "poisson_noisyB")
    pathC = os.path.join(path, "poisson_noisyC")
    os.makedirs(pathA, exist_ok=True)
    os.makedirs(pathB, exist_ok=True)
    os.makedirs(pathC, exist_ok=True)

    # 读取文件夹中的图片
    for root, dirs, files in os.walk(os.path.join(path, "high")):
        for file in files:
            if file.endswith(".png"):
                # Load the original image
                original_image = np.array(Image.open(os.path.join(root, file)))

                # Apply Poisson noise
                noisy_image = add_poisson_noise(original_image, scale=0.3)
                noisy_image2 = add_poisson_noise(original_image, scale=0.6)
                noisy_image3 = add_poisson_noise(original_image, scale=1.0)

                # 保存图片到本地
                noisy_image_pil = Image.fromarray(noisy_image)
                noisy_image_pil.save(os.path.join(pathA, file))
                noisy_image_pil = Image.fromarray(noisy_image2)
                noisy_image_pil.save(os.path.join(pathB, file))
                noisy_image_pil = Image.fromarray(noisy_image3)
                noisy_image_pil.save(os.path.join(pathC, file))


    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 4, 1)
    # plt.imshow(original_image, cmap='gray')
    # plt.title('Original Image')
    # plt.subplot(1, 4, 2)
    # plt.imshow(noisy_image, cmap='gray')
    # plt.title('Noisy Image (Poisson Noise)')
    # plt.subplot(1, 4, 3)
    # plt.imshow(noisy_image2, cmap='gray')
    # plt.title('Noisy Image (Poisson Noise)')
    # plt.subplot(1, 4, 4)
    # plt.imshow(noisy_image3, cmap='gray')
    # plt.title('Noisy Image (Poisson Noise)')
    # plt.show()
