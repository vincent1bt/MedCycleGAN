import os

import nibabel as nib
import numpy as np
import cv2

image_path = 'image_path'
output_folder = image_path[:-7]

os.mkdir(output_folder)

image_axis = 2
data = nib.load(image_path)
img = data.get_data()
shape = data.shape
final_size = 256

print(shape, "Shape")
print(img.min(), "Min value")
print(img.max(), "Max Value")

img = img / img.max()

img = img * 255

start = 50
end = 170

def get_square(image, square_size):
    height, width = image.shape    
    
    if(height > width):
      differ = height
    else:
      differ = width

    mask = np.zeros((differ, differ), dtype = "uint8")

    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)

    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]
    
    return mask

image_center = (final_size / 2, final_size / 2)
rotation_matrix = cv2.getRotationMatrix2D(image_center, 90, 1.0)

for index in range(start, end):
    img_2d = img[:, :, index]
    resized_img_2d = get_square(img_2d, final_size)
    rotated_img_2d = cv2.warpAffine(resized_img_2d, rotation_matrix, (final_size, final_size))

    cv2.imwrite(os.path.join(output_folder, f'{output_folder}-{index}.jpg'), rotated_img_2d)