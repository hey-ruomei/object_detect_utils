from flask import request, Flask, jsonify
from ultralytics import YOLO
from PIL import Image
from predict import predict_output
import time

import json
app = Flask(__name__)
model = YOLO("./runs/segment/train22/weights/best.pt")
@app.route("/")
def root():
    with open("index.html") as file:
         return file.read()
@app.route("/detect", methods=["POST"])
def detect():
     start_time = time.time()
     print(request.files)
     buf = request.files["image_file"]
     boxes = detect_objects_on_image(buf.stream)
     end_time = time.time()
     print(end_time - start_time)
     return jsonify(boxes)    
    
def detect_objects_on_image(buf):
     input_dir = buf
     return predict_output(input_dir, model)
if (__name__ == '__main__'):
    app.run(host="0.0.0.0",port=8003, threaded=False, debug=True)
