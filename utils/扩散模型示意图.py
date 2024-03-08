from PIL import Image
import numpy as np

# Load the image
image_path = r'/root/lmy/GanZoo/utils/手写数字.png'
original_image = Image.open(image_path).convert('L')  # Convert to grayscale

# Convert image to numpy array
original_image_np = np.array(original_image)


# Define a function to add noise to an image
def add_noise(img, noise_level):
    return img + noise_level * np.random.randn(*img.shape)


# Define a function to denoise an image by simple averaging filter (not optimal but simple for demonstration)
def denoise_simple_average(img, window_size=3):
    img_padded = np.pad(img, pad_width=window_size // 2, mode='constant', constant_values=0)
    denoised_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Create a window of specified size
            window = img_padded[i:i + window_size, j:j + window_size]
            # Calculate the mean of the window
            denoised_img[i, j] = np.mean(window)

    return denoised_img


# Add noise and denoise the image in three steps
noisy_images = []
denoised_images = [original_image_np.copy()]

for i in range(3):
    # Add noise
    noisy_image = add_noise(denoised_images[-1], noise_level=30)
    noisy_images.append(noisy_image)

    # Denoise
    denoised_image = denoise_simple_average(noisy_image)
    denoised_images.append(denoised_image)

# Plot the results
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

for i in range(3):
    # Original noisy image
    axs[i, 0].imshow(noisy_images[i], cmap='gray')
    axs[i, 0].axis('off')
    axs[i, 0].set_title(f'Noisy Step {i + 1}')

    # Denoised image
    axs[i, 1].imshow(denoised_images[i], cmap='gray')
    axs[i, 1].axis('off')
    axs[i, 1].set_title(f'Denoised Step {i + 1}')

# Set the last column to show the final denoised image
axs[0, 2].imshow(denoised_images[-1], cmap='gray')
axs[0, 2].axis('off')
axs[0, 2].set_title('Final Denoised Image')
# Hide the empty subplots
for i in range(1, 3):
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()
