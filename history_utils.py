"""
Local history for posture scans: save/load images and scores.
"""
import json
import os
import uuid
from datetime import datetime

HISTORY_DIR = "history"
META_FILE = "history_meta.json"


def ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _meta_path():
    return os.path.join(HISTORY_DIR, META_FILE)


def load_meta():
    ensure_history_dir()
    path = _meta_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_meta(entries):
    ensure_history_dir()
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def image_path(entry_id):
    return os.path.join(HISTORY_DIR, f"{entry_id}.jpg")


def save_scan(image_bytes, score, label, metrics_dict):
    """
    Save image and metadata. image_bytes: raw bytes (e.g. from upload).
    Returns the new entry dict including 'id'.
    """
    ensure_history_dir()
    meta = load_meta()
    entry_id = str(uuid.uuid4())
    img_path = image_path(entry_id)
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    entry = {
        "id": entry_id,
        "timestamp": datetime.now().isoformat(),
        "score": round(float(score), 1),
        "label": label,
        "metrics": {
            "shoulder_tilt_px": round(metrics_dict.get("shoulder_tilt_px", 0), 2),
            "hip_tilt_px": round(metrics_dict.get("hip_tilt_px", 0), 2),
            "torso_lean_deg": round(metrics_dict.get("torso_lean_deg", 0), 2),
        },
    }
    meta.append(entry)
    save_meta(meta)
    return entry


def get_entry_image_bytes(entry_id):
    path = image_path(entry_id)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def list_entries():
    """Return list of entries, newest first."""
    meta = load_meta()
    return sorted(meta, key=lambda e: e.get("timestamp", ""), reverse=True)


def get_entry_by_id(entry_id):
    for e in list_entries():
        if e.get("id") == entry_id:
            return e
    return None
