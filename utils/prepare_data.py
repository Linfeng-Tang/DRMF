import cv2
import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm

def IR_prepare(input_folder=None, output_folder=None, mean=0, stddev=25):
    os.makedirs(output_folder, exist_ok=True)
    filelist = natsorted(os.listdir(input_folder))
    file_bar = tqdm(filelist)
    for filename in file_bar:
        if filename.endswith((".jpg", ".png")):
            # Read the image
            img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

            # Add Gaussian noise
            noise = np.random.normal(mean, stddev, img.shape)
            noise = noise.astype(np.uint8)
            noisy_img = img + noise

            # Downsample by a factor of two
            downsampled_img = cv2.resize(noisy_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

            # Bicubic upsample to the original size
            bicubic_upsampled_img = cv2.resize(downsampled_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, bicubic_upsampled_img)

            file_bar.set_description(f"Processed: {filename}")

    print("Processing complete.")

def LOL_prepare(input_folder=None, output_folder=None, mean=0, stddev=25):
    os.makedirs(output_folder, exist_ok=True)
    filelist = natsorted(os.listdir(input_folder))
    file_bar = tqdm(filelist)
    for filename in file_bar:
        if filename.endswith((".jpg", ".png")):
            # Read the image
            img = cv2.imread(os.path.join(input_folder, filename))

            # Downsample by a factor of two
            downsampled_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            # Bicubic upsample to the original size
            bicubic_upsampled_img = cv2.resize(downsampled_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, bicubic_upsampled_img)

            file_bar.set_description(f"Processed: {filename}")

    print("Processing complete.")
    
if __name__ == '__main__':
    
    # Noise parameters
    mean = 0
    stddev = 5  # Standard deviation, controls noise intensity

    # Traverse images in the folder
    input_folder = "/data/timer/WeatherDiffusion/data/LLVIP/val/vi1"
    output_folder = "/data/timer/WeatherDiffusion/data/LLVIP/val/vi"
    LOL_prepare(input_folder=input_folder, output_folder=output_folder, mean=mean, stddev=stddev)
