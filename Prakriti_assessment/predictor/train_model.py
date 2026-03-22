import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

FACE_COLS = [
    "Shape of face",
    "Eyes",
    "Eyelashes",
    "Blinking of Eyes",
    "Cheeks",
    "Nose",
    "Lips",
    "Complexion",
]

SURVEY_COLS = [
    "Body Weight",
    "Appetite",
    "Bone Structure",
    "Appearance of Hair",
    "Body Size",
    "Teeth and gums",
    "Nails",
    "General feel of skin",
    "Height",
    "Texture of Skin",
    "Hair Color",
    "Liking tastes",
]


FEATURE_COLS = FACE_COLS + SURVEY_COLS
TARGET_COL   = "Dosha"

DEFAULT_MODEL_PATH = Path(__file__).parent / "model.pkl"
DEFAULT_DATA_PATH  = Path(__file__).parent / "data" / "Updated_Prakriti_With_Features.csv"

def load_csv(path: Path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_encoders(rows: list[dict], cols: list[str]) -> dict:
    """Fit one LabelEncoder per feature column."""
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        le.fit([r[col] for r in rows])
        encoders[col] = le
    return encoders


def encode_X(rows, encoders, feature_cols):
    X = []
    for r in rows:
        X.append([encoders[c].transform([r[c]])[0] for c in feature_cols])
    return np.array(X)


def train(data_path: Path, model_path: Path, test_size: float = 0.2,
          n_estimators: int = 200, random_state: int = 42) -> None:

    print(f"Loading data from: {data_path}")
    rows = load_csv(data_path)
    print(f"  {len(rows)} rows loaded")

    encoders = build_encoders(rows, FEATURE_COLS)
    X = encode_X(rows, encoders, FEATURE_COLS)

    label_enc = LabelEncoder()
    y = label_enc.fit_transform([r[TARGET_COL] for r in rows])
    print(f"  Classes: {list(label_enc.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print(f"\nTraining RandomForest (n_estimators={n_estimators})…")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

    artefact = {
        "clf":          clf,
        "encoders":     encoders,
        "label_enc":    label_enc,
        "feature_cols": FEATURE_COLS,
        "face_cols":    FACE_COLS,
        "survey_cols":  SURVEY_COLS,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Prakriti dosha classifier.")
    parser.add_argument("--data",   type=Path, default=DEFAULT_DATA_PATH,
                        help="Path to the CSV dataset")
    parser.add_argument("--model",  type=Path, default=DEFAULT_MODEL_PATH,
                        help="Where to save model.pkl")
    parser.add_argument("--trees",  type=int,  default=200,
                        help="Number of trees in the RandomForest (default 200)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use for testing (default 0.2)")
    parser.add_argument("--seed",   type=int,  default=42,
                        help="Random seed (default 42)")
    args = parser.parse_args()

    train(
        data_path    = args.data,
        model_path   = args.model,
        n_estimators = args.trees,
        test_size    = args.test_size,
        random_state = args.seed,
    )