# 一些做目标检测用的小工具
## 小目标检测
- 切割图片：small_object_cut_image.py

## 用 yolo 训练时用到的一些处理处理小工具
- detect 模型格式转换：detect_json2txt.py
- 验证标注框的准确性，将标注框重新画出来检查：show_box.py
- segmentation 模型格式转换：segmentation_json2txt
- 位样本手动添加水印：add_watermark.py
- 为样本添加摩尔纹：add_moire.py
- pdf批量转换图片：pdf2image.py

## 模型验证
- 测试接口：post.py
- 模型预测：predict.py
- 起本地接口调用模型进行预测：my_service.py
- 