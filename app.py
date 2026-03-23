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
from storage import save_assessment, get_all, delete_assessment, get_latest

DISEASE_MODEL_PATH = Path(__file__).parent / "Diagnosis" / "predictor" / "disease_model.pkl"
with open(DISEASE_MODEL_PATH, "rb") as f:
    _disease_model = pickle.load(f)

DOSHA_PREDISPOSITION = {
    "Vata":  {"Arthritis": 0.15, "Migraine": 0.10},
    "Pitta": {"Gastritis": 0.15, "Migraine": 0.10},
    "Kapha": {"Diarrhea":  0.10, "Arthritis": 0.08},
}

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


@app.route('/disease')
def disease():
    latest = get_latest()
    dosha  = latest['dosha'] if latest else None
    return render_template('disease.html', dosha=dosha)

@app.route('/wellness')
def wellness():
    latest = get_latest()
    dosha  = latest['dosha'] if latest else None
    return render_template('wellness.html', dosha=dosha, latest=latest)


@app.route('/profile')
def profile():
    records = get_all()
    return render_template('profile.html', records=records)


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


@app.route('/diagnose', methods=['POST'])
def diagnose():
    data        = request.json
    symptoms    = data.get("symptoms", [])
    severity    = data.get("severity", "LOW").upper()
    dosha_raw   = data.get("dosha", "")

    clf          = _disease_model['clf']
    le           = _disease_model['label_enc']
    sym_cols     = _disease_model['symptom_cols']
    drug_lookup  = _disease_model['drug_lookup']

    # Build feature vector
    vec = [1 if s in symptoms else 0 for s in sym_cols]

    # Base probabilities from model
    proba = clf.predict_proba([vec])[0].copy()
    classes = list(le.classes_)

    # Apply dosha weighting
    primary = dosha_raw.split('+')[0].capitalize() if dosha_raw else ""
    boosts  = DOSHA_PREDISPOSITION.get(primary, {})
    for disease, boost in boosts.items():
        if disease in classes:
            proba[classes.index(disease)] += boost

    # Normalize and predict
    proba = proba / proba.sum()
    pred_idx = int(proba.argmax())
    disease  = classes[pred_idx]
    confidence = round(float(proba[pred_idx]) * 100, 1)

    # Get medicines
    key = (disease.lower(), severity)
    medicines = drug_lookup.get(key, drug_lookup.get((disease.lower(), "NORMAL"), []))[:6]

    return jsonify({"disease": disease, "confidence": confidence, "medicines": medicines})


@app.route('/profile/delete/<record_id>', methods=['POST'])
def delete_record(record_id):
    ok = delete_assessment(record_id)
    return jsonify({"ok": ok})


if __name__ == '__main__':
    app.run(debug = True)
    