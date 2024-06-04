import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps

### 画出标注框
def drawPolyText(image, poly):
    for i in range(len(poly)):
        if i == len(poly) - 1:
            cv2.line(image, tuple(poly[i]), tuple(poly[0]), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.line(image, tuple(poly[i]), tuple(poly[i+1]), (0, 255, 0), 1, cv2.LINE_AA)
    return image

images_path = "../datasets/val/images"
labels_path = "../datasets/val/labels"
output_path = "../output_show"

for file_name in tqdm(os.listdir(labels_path)):
    image_file = os.path.join(images_path, file_name.replace(".txt", ".jpg"))
    image = Image.open(image_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image_width, image_height = image.size
    image = np.array(image)
    
    label_file = os.path.join(labels_path, file_name)
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splition = line.split("\n")[0].split(" ")[1:]
        poly = []
        for i in range(0, len(splition), 2):
            poly.append([int(float(splition[i]) * image_width), int(float(splition[i+1]) * image_height)])
        image = drawPolyText(image, poly)
    # TODO 不要转 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_path, file_name.replace(".txt", ".jpg")), image)