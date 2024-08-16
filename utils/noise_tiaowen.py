import os
import cv2
import numpy as np

def add_stripe(im):
    # img.shape = [h, w, c]
    stdN_G = np.random.uniform(25, 30)    # Control the intensity of Gaussian noise
    noise_G = np.random.normal(0, stdN_G, im.shape)
    beta = np.random.uniform(20, 25)  # Control the intensity of stripe noise
    noise_col = np.random.normal(0, beta, im.shape[1])
    S_noise = np.tile(noise_col, (im.shape[0], 1))
    return im + 1 * noise_G + 1 * S_noise

# Input and output folder paths
input_folder = "/data/timer/DRMF/data/LLVIP/val/ir_noise2"
output_folder = "/data/timer/DRMF/data/LLVIP/val/ir_noise"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # Read the original image
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        
        # Add random noise (e.g., Gaussian noise)
        noisy_image = add_stripe(img)
        
        # Save the noisy image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

print("Noise addition completed.")

