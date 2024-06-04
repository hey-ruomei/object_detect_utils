import cv2
import os

def crop_image(image, crop_size, overlap):
    img_height, img_width = image.shape[:2]
    crops = []
    step = crop_size - overlap
    for y in range(0, img_height, step):
        for x in range(0, img_width, step):
            # 剪切尺寸需要在图片尺寸范围内
            crop_x2 = min(x + crop_size, img_width)
            crop_y2 = min(y + crop_size, img_height)
            crop = image[y:crop_y2, x:crop_x2]
            # 传入实际剪切的宽高
            crops.append((crop, x, y, crop_x2 - x, crop_y2 - y))
    return crops

def update_annotations(annotations, x_offset, y_offset, crop_width, crop_height, img_width, img_height):
    updated_annotations = []
    for ann in annotations:
        class_id, x_center, y_center, width, height, = map(float, ann.split())
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height

        x1 = abs_x_center - abs_width / 2
        x2 = abs_x_center + abs_width / 2
        y1 = abs_y_center - abs_height / 2
        y2 = abs_y_center + abs_height / 2

        if (x_offset <= x1 < x_offset + crop_width) and (x_offset <= x2 < x_offset + crop_width) and \
           (y_offset <= y1 < y_offset + crop_height) and (y_offset <= y2 < y_offset + crop_height):
            new_x_center = (abs_x_center - x_offset) / crop_width
            new_y_center = (abs_y_center - y_offset) / crop_height
            new_width = abs_width / crop_width
            new_height = abs_height / crop_height
            updated_annotations.append(f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}")
    return updated_annotations

def process_image_and_annotations(image_path, annotations_path, crop_size, overlap, output_image_dir, output_labels_dir):
    image = cv2.imread(image_path)
    with open(annotations_path, 'r') as file:
        annotations = file.readlines()

    # 未切割的原图像尺寸
    img_height, img_width = image.shape[:2]
    crops = crop_image(image, crop_size, overlap)

    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(image_base_name)
    for i, (crop, x_offset, y_offset, crop_width, crop_height) in enumerate(crops):
        crop_height, crop_width = crop.shape[:2]
        updated_annotations = update_annotations(annotations, x_offset, y_offset, crop_width, crop_height, img_width, img_height)
        
        # 跳过没有标注的切块
        if not updated_annotations:
            continue

        crop_img_name = f"{image_base_name}_crop_{i}.jpg"
        cv2.imwrite(os.path.join(output_image_dir, crop_img_name), crop)

        crop_ann_name = f"{image_base_name}_crop_{i}.txt"
        with open(os.path.join(output_labels_dir, crop_ann_name), 'w') as file:
            file.write('\n'.join(updated_annotations))
            
def process_dataset(input_image_dir, input_labels_dir, output_image_dir, output_labels_dir, crop_size, overlap):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    image_files = [f for f in os.listdir(input_image_dir) if os.path.isfile(os.path.join(input_image_dir, f))]
    
    for image_file in image_files:
        image_path = os.path.join(input_image_dir, image_file)
        annotations_path = os.path.join(input_labels_dir, os.path.splitext(image_file)[0] + '.txt')

        if os.path.exists(annotations_path):
            process_image_and_annotations(image_path, annotations_path, crop_size, overlap, output_image_dir, output_labels_dir)
# 剪切验证集
input_image_dir_train = '../datasets/txt_data/0603_txt/val/images'
input_labels_dir_train = '../datasets/txt_data/0603_txt/val/labels'
output_image_dir_train = '../datasets/txt_data/0603_txt_cut/val/images'
output_labels_dir_train = '../datasets/txt_data/0603_txt_cut/val/labels'

crop_size = 512
overlap = 100  # 设置重叠区域大小

process_dataset(input_image_dir_train, input_labels_dir_train, output_image_dir_train, output_labels_dir_train, crop_size, overlap)


# 剪切训练集
input_image_dir_train = '../datasets/txt_data/0603_txt/train/images'
input_labels_dir_train = '../datasets/txt_data/0603_txt/train/labels'
output_image_dir_train = '../datasets/txt_data/0603_txt_cut/train/images'
output_labels_dir_train = '../datasets/txt_data/0603_txt_cut/train/labels'
process_dataset(input_image_dir_train, input_labels_dir_train, output_image_dir_train, output_labels_dir_train, crop_size, overlap)