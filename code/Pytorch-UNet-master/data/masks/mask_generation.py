# NOTE: this code was not designed to be portable. it was designed to be run once and never touched again
# if you want to modify it to make your own masks, CHANGE THE FILE PATHS. might need to change some other stuff too idk

from PIL import Image
import numpy as np
import os

dataset_path = "C:/Users/jonah/SoftDev/sign-detection/data/archive/car"
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
label_data = []
dark_mask_color = [200, 0, 200]
light_mask_color = [255, 255, 0]
image_counter = 0

with os.scandir(dataset_path) as entries:
    for entry in entries:
        if entry.is_dir():
            with os.scandir(entry.path + "/labels") as labels:
                for label in labels:
                    with open(label.path, 'r') as file:
                        label_data = file.read().split(' ')
                        if len(label_data) < 5:
                            print(label.path + " has less than 5 values")
                        else:
                            pixel_data = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.uint8)
                            center_x = label_data[1] * IMAGE_WIDTH
                            center_y = label_data[2] * IMAGE_HEIGHT
                            width = label_data[3] * IMAGE_WIDTH
                            height = label_data[4] * IMAGE_HEIGHT
                            for y in range(IMAGE_HEIGHT):
                                for x in range(IMAGE_WIDTH):
                                    # checking if the pixel is outside the ellipse indicated by the yolov8 label
                                    if ((x - center_x)**2 / (width/2)**2) + ((y - center_y)**2 / (height/2)**2) > 1:
                                        pixel_data[y, x] = light_mask_color
                                    else:
                                        pixel_data[y, x] = dark_mask_color
                            image = Image.fromarray(pixel_data, "RGB")
                            label_name = os.path.basename(file.name)
                            label_name = label_name[:(len(label_name) - 3)]
                            image_path = label.path + "/../masks/" + label_name + "png"
                            image.save(image_path)
                            image_counter += 1
                            print(f'generated {image_counter} masks')
                            