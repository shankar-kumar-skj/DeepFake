import os
import cv2
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("deepfake_celebdf_model.h5")

SEQ_LEN = 16

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < SEQ_LEN:
        return None

    idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames) / 255.0


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["video"]
        path = "uploaded.mp4"
        file.save(path)

        frames = extract_frames(path)

        if frames is None:
            result = "Video too short or unreadable"
        else:
            frames = np.expand_dims(frames, axis=0)
            prediction = model.predict(frames)[0][0]
            if prediction > 0.5:
                result = "REAL FACE ✔️"
            else:
                result = "FAKE / DEEPFAKE ❌"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
