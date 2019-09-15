import pydicom
import numpy as np
import cv2
import os

files = []

main_path = "file_path"

path = f"{main_path}/folder/CT"
output_path = f'{main_path}-axial'

os.mkdir(output_path)

for file_name in os.listdir(path):
    file = pydicom.read_file(f'{path}/{file_name}')
    files.append(file)

slices = []
skipcount = 0

for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

slices = sorted(slices, key=lambda s: s.SliceLocation)

img_shape = list(slices[0].pixel_array.shape)
print(img_shape)

img_shape.append(len(slices))
img3d = np.zeros(img_shape)

for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

min_bound = img3d.min()
max_bound = img3d.max()
    
def normalize(image):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1
    image[image < 0] = 0
    return image

img3d = normalize(img3d) * 255

img3d = img3d.astype(int)

img3d[(img3d > 80) & (img3d < 110)] = 0

for i in range(0, 233):
    cv2.imwrite(os.path.join(output_path, f'{main_path}-axial-{i}.jpg'), img3d[:, :, i])