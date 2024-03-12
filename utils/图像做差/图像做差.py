from PIL import Image
import numpy as np

# Load the images
image1 = Image.open('FD.png')
image2 = Image.open('QD.png')

# Convert the images to numpy arrays
image1_array = np.array(image1)
image2_array = np.array(image2)

# Calculate the difference
difference = np.abs(image1_array - image2_array)

# Convert the difference back to an image
difference_image = Image.fromarray(difference)

# Save the difference image
difference_image.save('diff.png')
