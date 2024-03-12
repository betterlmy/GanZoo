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


if __name__ == "__main__":

    # 创建文件夹
    path = r"/root/lmy/aapm512/high/"
    paths = [
        os.path.join(path, "16331.png"),
        os.path.join(path, "6.png"),
        os.path.join(path, "500.png"),
        os.path.join(path, "8000.png"),
        os.path.join(path, "4000.png"),
    ]
    os.makedirs("/root/lmy/GanZoo/utils/add_poi_noise/pics", exist_ok=True)
    for i, path in enumerate(paths):
        original_image = np.array(Image.open(path))
        scales = [0.93, 0.62, 0.86, 0.52, 0.42, 0.71]
        for scale in scales:
            noisy_image = add_poisson_noise(original_image, scale=scale)
            noisy_image_pil = Image.fromarray(noisy_image)
            noisy_image_pil.save(
                os.path.join(
                    "/root/lmy/GanZoo/utils/add_poi_noise/pics", f"{i}_{scale}.png"
                )
            )
            Image.fromarray(original_image).save(
                os.path.join("/root/lmy/GanZoo/utils/add_poi_noise/pics", f"{i}.png")
            )
