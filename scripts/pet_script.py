import os

import nibabel as nib
import numpy as np
import cv2

image_axis = 2

image_path = 'image_path'
output_folder = image_path[:-7]

os.mkdir(output_folder)

data = nib.load(image_path)
img = data.get_data()
shape = data.shape

print(shape, "Shape")
print(img.min(), "Min value")
print(img.max(), "Max Value")

img = img / img.max()

img = img * 255

start = 0
end = shape[image_axis]

alpha_channels = shape[3]

image_center = (shape[0] / 2, shape[1] / 2)
rotation_matrix = cv2.getRotationMatrix2D(image_center, 90, 1.0)

for index in range(start, end):
    img_2d = np.zeros((shape[0], shape[1]))

    for alpha_channel in range(0, alpha_channels):
        img_2d += img[:, :, index, alpha_channel] / alpha_channels
    
    img_2d = img_2d / img_2d.max()
    img_2d = img_2d * 255
    
    rotated_img_2d = cv2.warpAffine(img_2d, rotation_matrix, (shape[0], shape[1]))

    cv2.imwrite(os.path.join(output_folder, f'{output_folder}-{index}.jpg'), rotated_img_2d)