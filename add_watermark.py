from PIL import Image,ImageDraw,ImageFont,ImageEnhance
import os
import random
import json
import shutil
import cv2
import math

import numpy as np
'''
  批量为图片添加文字水印
'''
def randomtext():
    def GBK2312():
        head = random.randint(0xb0, 0xf7)
        body = random.randint(0xa1, 0xfe)
        val = f'{head:x}{body:x}'
        strr = bytes.fromhex(val).decode('gb2312',errors="ignore")
        return strr

    def randomABC():
        A = np.random.randint(65, 91)
        a = np.random.randint(97, 123)
        num = np.random.randint(48, 58)
        rand_len = np.random.randint(3, 15)
        char = ''.join([str(chr(n)) for n in np.random.choice([A, a, num], size=rand_len)])
        # print(char)
        return char

    rand_ratio = np.random.random()
    if rand_ratio<0.3:
        char = randomABC()
    else:

        char = ''
        rand_len = np.random.randint(3, 15)
        for i in range(rand_len):
            char = char + GBK2312()
    return char


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color
    # return "#D3D3D3"

# text = randomtext()
# # alphavalue = 0.4
# alphavalue = round(random.uniform(0.4, 0.6), 1)
# #设置所使用的字体
# font_size = random.randint(82, 142)
# font = ImageFont.truetype("./font/楷体_GB2312.ttf", font_size)
# 文字水印
def rotation_point(img, angle, pts):
    cols = img.shape[1]
    rows = img.shape[0]

    # 获取旋转变换矩阵
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # 计算旋转后的图像尺寸
    height_new = int(cols * math.fabs(math.sin(math.radians(angle))) + rows * math.fabs(math.cos(math.radians(angle)))) + 150
    width_new = int(rows * math.fabs(math.sin(math.radians(angle))) + cols * math.fabs(math.cos(math.radians(angle)))) + 150

    # 调整变换矩阵以进行平移
    M[0, 2] += (width_new - cols) / 2
    M[1, 2] += (height_new - rows) / 2

    # 对图像进行仿射变换（旋转）
    rotated_img = cv2.warpAffine(img, M, (width_new, height_new))

    # 对给定的点坐标进行仿射变换（旋转）
    pts_transformed = cv2.transform(np.asarray(pts, dtype=np.float64).reshape((-1, 1, 2)), M)

    return rotated_img, pts_transformed
def textMark(img, output_dir):
    text = randomtext()
    # alphavalue = 0.4
    alphavalue = round(random.uniform(0.4, 0.6), 1)
    #设置所使用的字体
    font_size = random.randint(52, 102)
    font = ImageFont.truetype("./font/楷体_GB2312.ttf", font_size)    
    
    try:
        im = Image.open(img).convert('RGBA') # 打开原始图片，并转换为RGB
        newImg = Image.new('RGBA', im.size, (255, 255, 255, 0)) # 存储添加水印后的图片
        imagedraw = ImageDraw.Draw(newImg) # 创建绘制对象
        imgwidth, imgheight = im.size # 记录图片大小
        print(font.getbbox(text))
        font_box = font.getbbox(text)
        txtwidth = font_box[2] - font_box[0] # 获取字体宽度
        txtheight = font_box[3] - font_box[1] # 获取字体高度
        # （0：左上角，1：左下角，2：右上角，3：右下角，4：居中）
        positionflag = [0, 1, 2, 3, 4][random.randint(0, 3)]
        # 设置水印文字位置
        if positionflag == 0: # 左上角
            position=(0 + 3 * txtheight,0 + 3 * txtheight)
        elif  positionflag == 1: # 左下角
             position=(0 + 3 * txtheight, imgheight - 3 * txtheight)
        elif  positionflag == 2: # 右上角
            position=(imgwidth - 1.5 * txtwidth,0+ 3 * txtheight)
        elif  positionflag == 3: # 右下角
            position=(imgwidth - 1.5 * txtwidth, imgheight - 3 * txtheight)
        elif  positionflag == 4: # 居中
            position=(imgwidth/2, imgheight/2)
        # 绘制文字
        imagedraw.text(position, text, font=font, fill=randomcolor())
        bbox = [position[0], position[1], position[0] + txtwidth, position[1] + txtheight]
        # 设置透明度
        alpha = newImg.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(alphavalue)
        newImg.putalpha(alpha)
        print(img)
        filename = img.split('/')[-1]
        Image.alpha_composite(im, newImg).save(output_dir + '/' + filename,'png') # 保存图片
        print(bbox)
        return bbox
    except Exception as e:
        print(e)

def add_water_main(input_dir, output_dir):
    
    try:
        list = os.listdir(input_dir)  # 遍历选择的文件夹
        for i in range(0, len(list)):  # 遍历文件列表
            filepath = os.path.join(input_dir, list[i])  # 记录遍历到的文件名
            if (filepath.endswith('.json')):
                print(filepath)
                with open(filepath, "r+", encoding="utf-8") as raw_annotation_file:
                    raw_annotation = json.loads(raw_annotation_file.read())
                    shapes = raw_annotation["shapes"]
                    has_watermark = False
                    # 判断标注文件中是否含有水印，需要筛选出无水印的图片添加水印及标注坐标
                    for shape in shapes:
                        if (shape["label"] == '21'):
                            has_watermark = True
                    
                    # 对应的图片文件path
                    image_path = filepath.replace('.json', '.jpg')
                    if (not os.path.exists(image_path)):
                        image_path = image_path.replace('.jpg', '.png')
                    if (not os.path.exists(image_path)):
                        image_path = image_path.replace('.png', '.jpeg')
                    print(f'image_path{image_path}')
                    if (not has_watermark):
                        box = textMark(image_path, output_dir) # 批量添加文字水印
                        bbox = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
                        temp = {
                            "label": "21",
                            "points": bbox,
                            "shape_type": "polygon"
                        }
                        temp_arr = raw_annotation["shapes"]
                        temp_arr.append(temp)
                        raw_annotation['shapes'] = temp_arr
                    else:
                        shutil.copy(image_path, output_dir + '/' + image_path.split('/')[1])
                    # 将更新后的数据写回 JSON 文件
                    target_path = output_dir + '/' + filepath.split('/')[-1]
                    with open(target_path, 'w') as json_file:
                        # 将数据写入 JSON 文件
                        json.dump(raw_annotation, json_file)            
        print('批量添加水印完成')
    except Exception as e:
            print(e)
if (__name__ == '__main__'):
    input_dir = 'test_watermark'
    output_dir = 'watermark_output'
    add_water_main(input_dir, output_dir)