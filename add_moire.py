import cv2
import random
import os
def choose_random_image(directory):
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在。")
        return None
    
    # 获取目录中所有图片文件的列表
    image_files = [file for file in os.listdir(directory) if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 检查是否找到图片文件
    if not image_files:
        print(f"警告：目录 '{directory}' 中没有找到图片文件。")
        return None
    
    # 从图片文件列表中随机选择一张图片
    random_image_file = random.choice(image_files)
    
    # 返回选中的图片文件路径
    return os.path.join(directory, random_image_file)
def add_moire(imagesDir, moire_dir, output_dir):
  try:
    list = os.listdir(imagesDir)  # 遍历选择的文件夹
    for i in range(0, len(list)):  # 遍历文件列表
        filepath = os.path.join(imagesDir, list[i])  # 记录遍历到的文件名
        raw = cv2.imread(filepath)
        height, width, _ = raw.shape
        ###摩尔纹掩码图
        mask_path = choose_random_image(moire_dir)
        mask = cv2.imread(mask_path)
        mask = cv2.resize(src=mask, dsize=(width, height))
        ###混合比例
        ratio = random.uniform(0.5, 0.9)
        image = raw * ratio + mask * (1 - ratio)
        cv2.imwrite(output_dir + '/' + list[i], image)
  except Exception as e:
          print(e)
if (__name__ == '__main__'):
  imagesDir = '../kazheng'
  moire_dir = '../mask'
  output_dir = '../mask_output'
  add_moire(imagesDir, moire_dir, output_dir)