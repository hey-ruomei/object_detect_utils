import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from io import BytesIO
import base64

def drawPoly(image, poly):
    for i in range(len(poly)):
        if i == len(poly) - 1:
            cv2.line(image, tuple(poly[i]), tuple(poly[0]), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.line(image, tuple(poly[i]), tuple(poly[i+1]), (0, 255, 0), 1, cv2.LINE_AA)
    return image
def predict_output(image, model):
    image = Image.open(image).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image_width, image_height = image.size
    results = model(image, verbose=False, device='cpu')
    result = results[0].masks
    image = np.array(image)
    if result:
        masks = result.data.cpu().numpy()
        _, mask_height, mask_width = masks.shape
        for mask in masks:
            mask = mask.astype(np.uint8)
            mask[mask != 0] = 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if contours:
                contours = contours[0]
                poly = []
                for contour in contours:
                    x, y = contour[0]
                    x = int(x * (image_width / mask_width))
                    y = int(y * (image_height / mask_height))
                    poly.append([x, y])
                image = drawPoly(image, poly)
    image_pil = Image.fromarray(image)
    image_stream = BytesIO()
    image_pil.save(image_stream, format="JPEG")
    # 将 BytesIO 对象中的数据转换为字节流
    image_bytes = image_stream.getvalue()
    # 将字节流编码为 base64 字符串（可选）
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    output = []
    max_text_box = []
    max_text_box_area = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        box_area = (x2 - x1) * (y2 - y1)
        # 文本框坐标只取最大框
        if (class_id == 1 and box_area > max_text_box_area):
            max_text_box_area = box_area
            max_text_box = [x1, y1, x2, y2, class_id, prob, image_base64]
        # 水印坐标直接 push
        if (class_id == 0):
            output.append([x1, y1, x2, y2, class_id, prob, image_base64])
    if (len(max_text_box) > 0):
        output.append(max_text_box)        
        # if (class_id == 1):
        #     box_name = '文本框'
        # else:
        #     box_name = '水印'
        # output.append([x1, y1, x2, y2, box_name, prob, image_base64])
    return output


def predict_directly(weight_dir, input_dir, output_dir):
    model = YOLO(weight_dir)
    image = Image.open(input_dir).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image_width, image_height = image.size
    # gpu
    # result = model(image, verbose=False, device='cuda')[0].masks
    # cpu
    result = model(image, verbose=False, device='cpu')[0].masks
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if result:
        masks = result.data.cpu().numpy()
        _, mask_height, mask_width = masks.shape
        for mask in masks:
            mask = mask.astype(np.uint8)
            mask[mask != 0] = 255
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if contours:
                contours = contours[0]
                poly = []
                for contour in contours:
                    x, y = contour[0]
                    x = int(x * (image_width / mask_width))
                    y = int(y * (image_height / mask_height))
                    poly.append([x, y])
                image = drawPoly(image, poly)
    cv2.imwrite(output_dir, image)
if (__name__ == '__main__'):
    weight_dir = "./runs/segment/train22/weights/best.pt"    
    input_dir = "./badcase/执照.jpg"
    output_dir = "./predict_output/3.png"
    model= YOLO(weight_dir)
    image = Image.open(input_dir).convert("RGB")
    image = ImageOps.exif_transpose(image)
    results = model(image, verbose=False, device='cpu')
    res = results[0].plot()
    cv2.imshow('result', res)
    cv2.waitKey(0)
    # predict_directly(weight_dir, input_dir, output_dir)
    