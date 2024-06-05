import cv2
import numpy as np
import torch
from torchvision.ops import nms

from ultralytics import YOLOv10
model = YOLOv10("./runs/detect/train4/weights/best.pt")
def crop_image(image, crop_size, overlap):
    img_height, img_width = image.shape[:2]
    crops = []
    step = crop_size - overlap
    
    for y in range(0, img_height, step):
        for x in range(0, img_width, step):
            crop_x2 = min(x + crop_size, img_width)
            crop_y2 = min(y + crop_size, img_height)
            crop = image[y:crop_y2, x:crop_x2]
            crops.append((crop, x, y))
    return crops

def draw_predictions(image, prediction):
    if (prediction[4] < 0.4): return image
    x1, y1, x2, y2, conf, cls = prediction[:6]
    label = f"类别：{cls} 置信度：{conf:.2f}"
    color = (255, 0, 0)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image

def detect_objects_in_cropped_images(image_path, crop_size, overlap):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    
    crops = crop_image(image, crop_size, overlap)
    all_predictions = np.empty((0, 6), int)
    
    for crop, x_offset, y_offset in crops:
        results = model(crop)
        predictions = results[0]
        if (predictions):
            xyxy_preds = predictions.boxes.xyxy
            for index, pred in enumerate(xyxy_preds):
                x1 = float(pred[0]) + x_offset
                y1 = float(pred[1]) + y_offset
                x2 = float(pred[2]) + x_offset
                y2 = float(pred[3]) + y_offset
                conf = float(predictions.boxes.conf[index])
                cls = float(predictions.boxes.cls[index])
                all_predictions = np.vstack((all_predictions, [x1, y1, x2, y2, conf, cls]))        
    
    # 使用非极大值抑制去重
    boxes = all_predictions[:, :4]  # 预测框的坐标
    scores = all_predictions[:, 4]  # 置信度
    iou_threshold = 0.5
    indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
    final_predictions = all_predictions[indices]
    
    # 只有一个框或者无重叠框的情况 final_predictions 会是一个一维数组，需要统一处理成二维的
    if (final_predictions.ndim < 2):
        final_predictions = [final_predictions]
        
    for predictions in final_predictions:
        print('----------', predictions)
        image = draw_predictions(image, predictions)
    
    return image

# 处理图像
image_path = 'image_path'

output_image_path = './show2.jpg'
crop_size = 320
overlap = 100

result_image = detect_objects_in_cropped_images(image_path, crop_size, overlap)
cv2.imwrite(output_image_path, result_image)