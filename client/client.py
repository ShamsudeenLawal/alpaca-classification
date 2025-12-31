# client code for testing
import os

images_files = os.listdir("client/test-images")

for image_file in images_files:
    import requests
    with open(f"client/test-images/{image_file}", "rb") as f:
        files = {"file": (image_file, f, "image/jpeg")}
        response = requests.post(url="http://localhost:8000/predict", files=files)
        print(f"Image: {image_file}, Prediction: {response.json()}")