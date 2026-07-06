"""
Lightweight face engine.
Replaces face_arcface.py + yolo_detector.py + faiss_db.py with a single class.

- Detection AND embedding both come from insightface (no separate YOLO model needed).
- Model pack is configurable via FACE_MODEL env var:
    "buffalo_l"  -> best accuracy (~330MB, downloaded once at first run)
    "buffalo_s"  -> smaller (~50MB), still good accuracy for classroom-size datasets
  Default: buffalo_l (best accuracy). Change to buffalo_s if you need a smaller
  runtime footprint (e.g. very constrained free-tier disk).
"""
import os
import numpy as np
from insightface.app import FaceAnalysis

MODEL_NAME = os.environ.get("FACE_MODEL", "buffalo_s")
DET_SIZE = int(os.environ.get("FACE_DET_SIZE", "320"))


class FaceEngine:
    _instance = None

    def __init__(self):
        self.app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))

    @classmethod
    def instance(cls):
        """Singleton so the model loads only once (important for gunicorn workers=1)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_faces(self, img_rgb):
        """
        Detect all faces in an RGB numpy image.
        Returns list of (box, embedding) where box = (x1, y1, x2, y2)
        and embedding is a unit-normalized 512-d vector.
        """
        faces = self.app.get(img_rgb)
        results = []
        for f in faces:
            emb = f.embedding
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = emb / norm
            x1, y1, x2, y2 = f.bbox.astype(int)
            results.append(((int(x1), int(y1), int(x2), int(y2)), emb))
        return results

    def get_embedding(self, img_rgb):
        """Convenience method: returns embedding of the first/best face found."""
        faces = self.app.get(img_rgb)
        if len(faces) == 0:
            raise ValueError("No face found in image")
        emb = faces[0].embedding
        return emb / np.linalg.norm(emb)
