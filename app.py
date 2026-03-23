import base64
import pickle
import numpy as np
import io
from pathlib import Path
from PIL import Image
from flask import Flask, render_template, request, jsonify

from Prakriti_assessment.vision_model.face_features import _extract
from storage import save_assessment, get_all, delete_assessment, get_latest

app = Flask(__name__)
app.secret_key = "vedapath-secret-2026"

#Load Prakriti model 
MODEL_PATH = Path(__file__).parent / "Prakriti_assessment" / "predictor" / "model.pkl"
with open(MODEL_PATH, "rb") as f:
    _model = pickle.load(f)

#Load Disease model
DISEASE_MODEL_PATH = Path(__file__).parent / "Diagnosis" / "predictor" / "disease_model.pkl"
with open(DISEASE_MODEL_PATH, "rb") as f:
    _disease_model = pickle.load(f)

# Dosha constitutional predispositions — boosts probability for prone diseases
DOSHA_PREDISPOSITION = {
    "Vata":  {"Arthritis": 0.12, "Migraine": 0.10, "Cervical spondylosis": 0.10,
              "Paralysis (brain hemorrhage)": 0.08},
    "Pitta": {"GERD": 0.12, "Peptic ulcer diseae": 0.10, "Hypertension": 0.10,
              "Heart attack": 0.08, "Hepatitis B": 0.08},
    "Kapha": {"Diabetes": 0.12, "Obesity (not in dataset)": 0.10,
              "Hypothyroidism": 0.10, "Bronchial Asthma": 0.08, "Pneumonia": 0.08},
}


#Pages 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prakriti')
def prakriti():
    return render_template('prakriti.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/profile')
def profile():
    records = get_all()
    return render_template('profile.html', records=records)

@app.route('/wellness')
def wellness():
    latest = get_latest()
    dosha  = latest['dosha'] if latest else None
    return render_template('wellness.html', dosha=dosha, latest=latest)

@app.route('/disease')
def disease():
    latest = get_latest()
    dosha  = latest['dosha'] if latest else None
    return render_template('disease.html', dosha=dosha)


#API

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
    data      = request.json
    clf       = _model['clf']
    encoders  = _model['encoders']
    label_enc = _model['label_enc']
    feat_cols = _model['feature_cols']
    face_cols = _model.get('face_cols', [])

    row, missing = [], []
    for col in feat_cols:
        val = data.get(col)
        if val is None:
            missing.append(col)
            continue
        try:
            row.append(encoders[col].transform([val])[0])
        except ValueError:
            row.append(0)

    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    pred       = clf.predict([row])[0]
    dosha      = label_enc.inverse_transform([pred])[0]
    confidence = round(float(clf.predict_proba([row])[0].max()) * 100, 1)

    # Save to profile history
    face_features  = {k: v for k, v in data.items() if k in face_cols}
    survey_answers = {k: v for k, v in data.items() if k not in face_cols}
    save_assessment(dosha, confidence, face_features, survey_answers)

    return jsonify({"dosha": dosha, "confidence": confidence})


@app.route('/diagnose', methods=['POST'])
def diagnose():
    data      = request.json
    symptoms  = data.get("symptoms", [])
    dosha_raw = data.get("dosha", "")

    clf      = _disease_model['clf']
    le       = _disease_model['label_enc']
    sym_cols = _disease_model['symptom_cols']
    ayur_kb  = _disease_model['ayur_kb']

    # Build binary feature vector
    vec   = [1 if s in symptoms else 0 for s in sym_cols]
    proba = clf.predict_proba([vec])[0].copy()
    classes = list(le.classes_)

    # Apply dosha constitutional weighting
    primary = dosha_raw.split('+')[0].strip().capitalize() if dosha_raw else ""
    for disease, boost in DOSHA_PREDISPOSITION.get(primary, {}).items():
        if disease in classes:
            proba[classes.index(disease)] += boost

    proba      = proba / proba.sum()
    pred_idx   = int(proba.argmax())
    disease    = classes[pred_idx]
    confidence = round(float(proba[pred_idx]) * 100, 1)

    # Fetch Ayurvedic info from knowledge base
    info = ayur_kb.get(disease.lower(), {})

    return jsonify({
        "disease":       disease,
        "confidence":    confidence,
        "hindi":         info.get("hindi", ""),
        "doshas":        info.get("doshas", ""),
        "herbs":         info.get("herbs", ""),
        "formulation":   info.get("formulation", ""),
        "diet":          info.get("diet", ""),
        "yoga":          info.get("yoga", ""),
        "prevention":    info.get("prevention", ""),
        "severity":      info.get("severity", ""),
        "prognosis":     info.get("prognosis", ""),
        "complications": info.get("complications", ""),
    })


@app.route('/profile/delete/<record_id>', methods=['POST'])
def delete_record(record_id):
    ok = delete_assessment(record_id)
    return jsonify({"ok": ok})


if __name__ == '__main__':
    app.run(debug=True)
