import torch
import torchvision.transforms as transforms
from PIL import Image

# Load your original image and convert it to a tensor
# Replace 'your_image_path.png' with the path to your image
image_path = '../B301MM/high/LD1.png'
with Image.open(image_path) as img:
    img_gray = img.convert('L')  # Convert to grayscale
transform_to_tensor = transforms.ToTensor()
img_tensor = transform_to_tensor(img_gray)


# Function to create noisy images with different beta_end values
def create_noisy_images_with_beta_end(beta_end, T=1000, timesteps=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]):
    beta_schedule = torch.linspace(0.0001, beta_end, T)
    noisy_images = [img_tensor]  # Start with the original image tensor

    for t in timesteps:
        beta_t = beta_schedule[t - 1]  # Adjust for zero-based indexing
        noise = torch.randn_like(img_tensor)
        noisy_img = torch.sqrt(1 - beta_t) * img_tensor + torch.sqrt(beta_t) * noise
        noisy_images.append(noisy_img)

    concatenated_image = torch.cat(noisy_images, dim=2)
    return concatenated_image


# Define different beta_end values to test
beta_end_values = [0.02]

# Create noisy images for each beta_end value
noisy_images_for_beta_ends = {beta_end: create_noisy_images_with_beta_end(beta_end) for beta_end in beta_end_values}

# Convert concatenated tensors to PIL images for visualization
to_pil = transforms.ToPILImage()
concatenated_images_pil_for_beta_ends = {beta_end: to_pil(img) for beta_end, img in noisy_images_for_beta_ends.items()}

# Save or display the images for each beta_end value
for beta_end, img_pil in concatenated_images_pil_for_beta_ends.items():
    img_pil.save(f"add_noise_test/noisy_images_beta_end_{beta_end}.png")  # Save the image
    img_pil.show()  # Display the image
