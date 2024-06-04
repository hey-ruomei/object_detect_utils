import requests
import base64
image_path = './badcase/20240320140335529.jpg'
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_str
image = image_to_base64(image_path)
files = {
    "image": image
}
data = {
    "requestId": '123456',
    "image": image
}
json_data = requests.post('your url', data=data).json()
print(json_data)