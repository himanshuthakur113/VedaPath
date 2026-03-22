import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"


LM = {
    "top": 10, "chin": 152,
    "left_cheek": 234, "right_cheek": 454,
    "left_jaw": 172, "right_jaw": 397,
    "l_eye_top": 159, "l_eye_bot": 145, "l_eye_left": 33, "l_eye_right": 133,
    "r_eye_top": 386, "r_eye_bot": 374, "r_eye_left": 362, "r_eye_right": 263,
    "l_upper_lid": 160, "r_upper_lid": 387,
    "nose_left": 129, "nose_right": 358,
    "lip_left": 61, "lip_right": 291,
    "l_cheekbone": 116, "r_cheekbone": 345,
    "l_cheek_mid": 205, "r_cheek_mid": 425,
}


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _landmarks(rgb):
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opts) as d:
        result = d.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return result.face_landmarks[0] if result.face_landmarks else None


def _extract(rgb):
    lm = _landmarks(rgb)
    if lm is None:
        return None

    face_w = _dist(lm[LM["left_cheek"]], lm[LM["right_cheek"]])
    face_h = _dist(lm[LM["top"]], lm[LM["chin"]])
    eye_w  = (_dist(lm[LM["l_eye_left"]], lm[LM["l_eye_right"]]) +
              _dist(lm[LM["r_eye_left"]], lm[LM["r_eye_right"]])) / 2
    eye_h  = (_dist(lm[LM["l_eye_top"]], lm[LM["l_eye_bot"]]) +
              _dist(lm[LM["r_eye_top"]], lm[LM["r_eye_bot"]])) / 2

    # 1. Shape of face
    hw    = face_h / face_w if face_w else 0
    taper = _dist(lm[LM["left_jaw"]], lm[LM["right_jaw"]]) / face_w if face_w else 0
    if hw > 1.45:            shape_of_face = "Long, angular, thin"
    elif taper < 0.70:       shape_of_face = "Heart-shaped, pointed chin"
    else:                    shape_of_face = "Large, round, full"

    # 2. Eyes
    eye_ratio = eye_w / face_w if face_w else 0
    if eye_ratio < 0.14:     eyes = "Small, active, darting, dark eyes"
    elif eye_ratio < 0.19:   eyes = "Medium-sized, penetrating, light-sensitive eyes"
    else:                    eyes = "Big, round, beautiful, glowing eyes"

    # 3. Eyelashes (upper lid height proxy)
    lid_h = (_dist(lm[LM["l_upper_lid"]], lm[LM["l_eye_bot"]]) +
             _dist(lm[LM["r_upper_lid"]], lm[LM["r_eye_bot"]])) / 2
    lid_ratio = lid_h / face_h if face_h else 0
    if lid_ratio < 0.025:    eyelashes = "Scanty eyelashes"
    elif lid_ratio < 0.04:   eyelashes = "Moderate eyelashes"
    else:                    eyelashes = "Thick/Fused eyelashes"

    # 4. Blinking (eye aspect ratio)
    ear = eye_h / eye_w if eye_w else 0
    if ear < 0.22:           blinking = "Excessive Blinking"
    elif ear < 0.32:         blinking = "Moderate Blinking"
    else:                    blinking = "More or less stable"

    # 5. Cheeks
    cheek_ratio = _dist(lm[LM["l_cheekbone"]], lm[LM["r_cheekbone"]]) / face_w if face_w else 0
    if cheek_ratio < 0.75:   cheeks = "Wrinkled, Sunken"
    elif cheek_ratio < 0.88: cheeks = "Smooth, Flat"
    else:                    cheeks = "Rounded, Plump"

    # 6. Nose
    nose_ratio = _dist(lm[LM["nose_left"]], lm[LM["nose_right"]]) / face_w if face_w else 0
    if nose_ratio < 0.22:    nose = "Crooked, Narrow"
    elif nose_ratio < 0.28:  nose = "Pointed, Average"
    else:                    nose = "Rounded, Large open nostrils"

    # 7. Lips
    lip_ratio = _dist(lm[LM["lip_left"]], lm[LM["lip_right"]]) / face_w if face_w else 0
    if lip_ratio < 0.28:     lips = "Tight, thin, dry lips which chaps easily"
    elif lip_ratio < 0.36:   lips = "Lips are soft, medium-sized"
    else:                    lips = "Lips are large, soft, pink, and full"

    # 8. Complexion (sample skin from both cheeks)
    h, w = rgb.shape[:2]
    samples = []
    for key in ("l_cheek_mid", "r_cheek_mid"):
        cx, cy = int(lm[LM[key]].x * w), int(lm[LM[key]].y * h)
        patch  = rgb[max(0, cy-12):cy+12, max(0, cx-12):cx+12]
        if patch.size:
            samples.append(patch.mean(axis=(0, 1)))
    r, g, b = np.mean(samples, axis=0) if samples else (180, 150, 130)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > 200:            complexion = "White, pale, tans easily"
    elif lum > 150:          complexion = "Fair-skin sunburns easily"
    else:                    complexion = "Dark-Complexion, tans easily"

    return {
        "Shape of face":    shape_of_face,
        "Eyes":             eyes,
        "Eyelashes":        eyelashes,
        "Blinking of Eyes": blinking,
        "Cheeks":           cheeks,
        "Nose":             nose,
        "Lips":             lips,
        "Complexion":       complexion,
    }


def extract_from_path(image_path: str) -> dict | None:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    return _extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def extract_from_webcam(camera_index: int = 0) -> dict | None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    result = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, "SPACE = capture  |  Q = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
        cv2.imshow("Prakriti Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            result = _extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result is None:
                cv2.putText(frame, "No face — try again",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2)
                cv2.imshow("Prakriti Capture", frame)
                cv2.waitKey(1000)
                continue
            break

    cap.release()
    cv2.destroyAllWindows()
    return result
