import base64
import pickle
import numpy as np
import io
from pathlib import Path
from PIL import Image

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from Prakriti_assessment.vision_model.face_features import _extract

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "Prakriti_assessment" / "predictor" / "model.pkl"
with open(MODEL_PATH, "rb") as f:
    _model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prakriti')
def prakriti():
    return render_template('prakriti.html')


@app.route('/survey')
def survey():
    return render_template('survey.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "No image data"}), 400

    header, encoded = image_b64.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    results = _extract(rgb)
    if results is None:
        return jsonify({"error": "No face detected — please try again"}), 400

    return jsonify(results)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    clf       = _model['clf']
    encoders  = _model['encoders']
    label_enc = _model['label_enc']
    feat_cols = _model['feature_cols']

    row = []
    missing = []
    for col in feat_cols:
        val = data.get(col)
        if val is None:
            missing.append(col)
            continue
        try:
            encoded = encoders[col].transform([val])[0]
            row.append(encoded)
        except ValueError:
            # unseen label — use nearest class by index
            classes = list(encoders[col].classes_)
            row.append(0)

    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    pred = clf.predict([row])[0]
    dosha = label_enc.inverse_transform([pred])[0]
    proba = clf.predict_proba([row])[0]
    confidence = round(float(proba.max()) * 100, 1)

    return jsonify({"dosha": dosha, "confidence": confidence})


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
