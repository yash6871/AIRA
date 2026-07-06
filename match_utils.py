"""
Replaces faiss_db.py. For attendance-system scale (tens to a few thousand
students) plain numpy cosine similarity is just as fast as faiss and removes
a ~50MB dependency plus the whole faiss build headache.
"""
import numpy as np


def find_best_match(emb, embeddings, labels, threshold=0.5):
    """
    emb: unit-normalized query embedding (np.array, shape (512,))
    embeddings: list/array of unit-normalized stored embeddings
    labels: list of labels (e.g. (name, roll)) parallel to embeddings
    Returns: (label_or_None, score)
    """
    if not embeddings:
        return None, 0.0

    mat = np.array(embeddings, dtype="float32")
    sims = mat @ emb  # cosine similarity since all vectors are unit-normalized
    idx = int(np.argmax(sims))
    score = float(sims[idx])

    if score >= threshold:
        return labels[idx], score
    return None, score


def decode_base64_image(data_url):
    """Decode a data URL (e.g. from <canvas>.toDataURL()) into an RGB numpy array."""
    import base64
    import io
    from PIL import Image

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)
