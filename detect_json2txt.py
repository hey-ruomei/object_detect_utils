import os
import json
import random
from tqdm import tqdm
from PIL import Image, ImageOps

# raw_path = "../test_json2txt"       ### labelme数据集路径
raw_path = "../origin_data/0603_check_list_origin"       ### labelme数据集路径

# raw_path = "./hsv_transformed_output"       ### labelme数据集路径
yolo_train_images_path = "../datasets/txt_data/0603_txt/train/images"
yolo_train_labels_path = "../datasets/txt_data/0603_txt/train/labels"
yolo_val_images_path = "../datasets/txt_data/0603_txt/val/images"
yolo_val_labels_path = "../datasets/txt_data/0603_txt/val/labels"
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
    if file_name.endswith(".jpg") or file_name.endswith('.png') or file_name.endswith('.jpeg'):
        raw_image_path = os.path.join(raw_path, file_name)
        pre_file_name = file_name.rsplit('.', 1)[0]
        raw_annotation_path = os.path.join(raw_path, pre_file_name + '.json')
        
        image = Image.open(raw_image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image_width, image_height = image.size
        
        json_abs_path = os.path.abspath(raw_annotation_path)
        if (not os.path.exists(json_abs_path)):
          print(f"{json_abs_path} not exits")
          continue
        with open(raw_annotation_path, "r", encoding="utf-8") as raw_annotation_file:
            raw_annotation = json.loads(raw_annotation_file.read())
        
        ### 拆分训练集验证集
        if random.random() <= 0.97:
            image.save(os.path.join(yolo_train_images_path, "train_" + pre_file_name + "_" + str(train_index) + ".jpg"))
            yolo_label_file = open(os.path.join(yolo_train_labels_path, "train_" + pre_file_name + "_" + str(train_index) + ".txt"), "w")
            train_index += 1
        else:
            image.save(os.path.join(yolo_val_images_path, "val_"  + pre_file_name + "_" + str(val_index) + ".jpg"))
            yolo_label_file = open(os.path.join(yolo_val_labels_path, "val_" + pre_file_name + "_" + str(val_index) + ".txt"), "w")
            val_index += 1
        
        ### 生成标注文件
        shapes = raw_annotation["shapes"]
        for shape in shapes:
            poly = ["500"]
            points = shape["points"]
            if (shape["label"] == '0'):
                poly = ["0"]
            elif (shape["label"] == '1'):
                poly = ["1"]
            elif (shape["label"] == '2'):
                poly = ["2"]
            elif (shape["label"] == '3'):
                poly = ["3"]
            elif (shape["label"] == '4'):
                poly = ["4"]
            elif (shape["label"] == '5'):
                poly = ["5"]
            elif (shape["label"] == '6'):
                poly = ["6"]
            elif (shape["label"] == '7'):
                poly = ["7"]
            elif (shape["label"] == '8'):
                poly = ["8"]
            elif (shape["label"] == '9'):
                poly = ["9"]
            cx = (points[1][0] + points[0][0]) / 2 / image_width
            cy = (points[1][1] + points[0][1]) / 2 / image_height
            width = (points[1][0] - points[0][0]) / image_width
            height = (points[1][1] - points[0][1]) / image_height
            poly.extend([str(cx), str(cy), str(width), str(height)])
            yolo_label_file.write(" ".join(poly) + "\n")
        yolo_label_file.close()