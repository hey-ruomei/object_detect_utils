import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageOps

raw_path = "../0523_train_add_moire"       ### labelme数据集路径
# raw_path = "./hsv_transformed_output"       ### labelme数据集路径
yolo_train_images_path = "../0523_txt/train/images"
yolo_train_labels_path = "../0523_txt/train/labels"
yolo_val_images_path = "../0523_txt/val/images"
yolo_val_labels_path = "../0523_txt/val/labels"
train_index = 1
val_index = 1

def create_nested_folder(relative_path):
    # 获取绝对路径
    abs_path = os.path.abspath(relative_path)
    
    # 如果文件夹已经存在，则直接返回
    if os.path.exists(abs_path):
        return
    
    # 创建文件夹
    os.makedirs(abs_path)
create_nested_folder(yolo_train_images_path)
create_nested_folder(yolo_train_labels_path)
create_nested_folder(yolo_val_images_path)
create_nested_folder(yolo_val_labels_path)

for file_name in tqdm(os.listdir(raw_path)):
    if file_name.endswith(".jpg") or file_name.endswith('.png'):
        raw_image_path = os.path.join(raw_path, file_name)
        pre_file_name = file_name.split('.')[0]
        raw_annotation_path = os.path.join(raw_path, pre_file_name + '.json')
        
        image = Image.open(raw_image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image_width, image_height = image.size
        
        with open(raw_annotation_path, "r", encoding="utf-8") as raw_annotation_file:
            raw_annotation = json.loads(raw_annotation_file.read())
        
        ### 拆分训练集验证集
        if random.random() <= 0.97:
            image.save(os.path.join(yolo_train_images_path, "train_" + str(train_index) + ".jpg"))
            yolo_label_file = open(os.path.join(yolo_train_labels_path, "train_" + str(train_index) + ".txt"), "w")
            train_index += 1
        else:
            image.save(os.path.join(yolo_val_images_path, "val_" + str(val_index) + ".jpg"))
            yolo_label_file = open(os.path.join(yolo_val_labels_path, "val_" + str(val_index) + ".txt"), "w")
            val_index += 1
        
        ### 生成标注文件
        shapes = raw_annotation["shapes"]
        for shape in shapes:
            points = shape["points"]
            if (shape["label"] == '21'):
                poly = ["0"]
            else:
                poly = ["1"]
            for point in points:
                poly.append(str(point[0] / image_width))
                poly.append(str(point[1] / image_height))
            yolo_label_file.write(" ".join(poly) + "\n")
        yolo_label_file.close()