# server.py
import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

from model_loader import load_deepfake_model

# CONFIG
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"mp4", "mov", "avi", "mkv"}
MAX_FILE_SIZE_MB = 200   # limit upload file size (adjust as needed)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

# Load model once at startup
print("[INFO] Loading model...")
model, SEQ_LEN, IMG_SIZE = load_deepfake_model()  # uses defaults, or pass different paths
print("[INFO] Model loaded.")

# Helper functions
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return '.' in filename and ext in ALLOWED_EXT

def extract_frames_on_the_fly(video_path, seq_len=16, img_size=224):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < seq_len:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, seq_len).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32)

def predict_video(video_path):
    frames = extract_frames_on_the_fly(video_path, seq_len=SEQ_LEN, img_size=IMG_SIZE)
    if frames is None:
        return {"ok": False, "error": "Video too short or unreadable (shorter than seq_len)."}
    x = frames / 255.0
    x = np.expand_dims(x, axis=0)  # shape (1, seq_len, h, w, 3)
    pred = model.predict(x, verbose=0)[0][0]
    label = "real" if float(pred) >= 0.5 else "fake"
    confidence = float(pred) if label == "real" else float(1 - pred)
    return {"ok": True, "label": label, "score": float(pred), "confidence": confidence}

# HTML upload page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", error="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(save_path)
            # predict
            res = predict_video(save_path)
            # cleanup uploaded file
            try:
                os.remove(save_path)
            except Exception:
                pass
            if not res['ok']:
                return render_template("index.html", error=res.get("error"))
            return render_template("index.html", result=res)
        else:
            return render_template("index.html", error="File type not allowed")
    return render_template("index.html")

# Simple JSON API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)

    res = predict_video(save_path)
    try:
        os.remove(save_path)
    except Exception:
        pass

    if not res['ok']:
        return jsonify(res), 400
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
