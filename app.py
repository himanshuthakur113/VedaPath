import base64
import numpy as np
import io
from PIL import Image
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from Prakriti_assessment.vision_model.face_features import _extract

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prakriti')
def prakriti():
    return render_template('prakriti.html')

@app.route('/analyze',methods=['POST'])
def analyze():
    data = request.json
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"error": "No image data"}), 400
    
    header, encoded = image_b64.split(",",1)
    image_bytes = base64.b64decode(encoded)
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    results = _extract(rgb)

    return jsonify(results)


@app.route('/disease')
def disease():
    return "<h1>Disease Detection Page</h1>"

@app.route('/wellness')
def wellness():
    return "<h1>General Wellness Page</h1>"

@app.route('/profile')
def profile():
    return "<h1>User Profile Page</h1>"

if __name__ == '__main__':
    app.run(debug = True)
