import json
import uuid
from datetime import datetime
from pathlib import Path

STORE = Path(__file__).parent / "data" / "assessments.json"

def _load() -> list:
    if not STORE.exists():
        return []
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    

def _save(records: list) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def save_assessment(dosha: str, confidence: float,
                    face_features: dict, survey_answers: dict) -> str:
    """Persist one assessment. Returns the new record's id."""
    records = _load()
    record = {
        "id":             str(uuid.uuid4()),
        "date":           datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "dosha":          dosha,
        "confidence":     confidence,
        "face_features":  face_features,
        "survey_answers": survey_answers,
    }
    records.insert(0, record)   # newest first
    _save(records)
    return record["id"]


def get_all() -> list:
    return _load()


def delete_assessment(record_id: str) -> bool:
    records = _load()
    new = [r for r in records if r["id"] != record_id]
    if len(new) == len(records):
        return False
    _save(new)
    return True


def get_latest() -> dict | None:
    records = _load()
    return records[0] if records else None

