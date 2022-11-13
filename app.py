from flask import Flask, request
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
app = Flask(__name__)

MODEL = tf.keras.models.load_model(
    "./_function InceptionV3 at 0x7f1b06a7f7a0__1.h5")
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route('/')
def home():
    return "hello world"

@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == 'POST':
        file = request.files['img']
        image = read_file_as_image(file.read())
        output = cv2.resize(image, (224, 224))
        img_batch = np.expand_dims(output, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    else:
        return "Method not allowed"
app.run()
