import os
import cv2
import numpy as np

# Input and output folder paths
input_folder = "/data/timer/DRMF/data/LLVIP/test/vi"
output_folder = "/data/timer/DRMF/data/LLVIP/test/vi_noise"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # Read the original image
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        print(filename)
        # Add random noise (e.g., Gaussian noise)
        noise_level = np.random.randint(5, 10)
        noise = np.random.normal(0, 5, img.shape)
        noisy_image = img + noise
        
        # Set the gamma value (less than 1 darkens the image, greater than 1 brightens the image)
        gamma = 1.2

        # Apply gamma correction to the image
        gamma_corrected = np.power(noisy_image/255.0, gamma)
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        # Save the noisy image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

print("Noise addition completed.")
