# helpers/video_processing.py

import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from .constants import VIDEO_FOLDER, IMG_SIZE, SEQUENCE_LENGTH, CLASS_MAP, cnn_model, video_model

def get_uploaded_file(file_key):
    from flask import request, jsonify
    
    if file_key not in request.files:
        return None, jsonify({"error": f"No {file_key} uploaded"}), 400
    
    file = request.files[file_key]
    if file.filename == '':
        return None, jsonify({"error": f"No selected {file_key} file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(VIDEO_FOLDER, filename)
    file.save(file_path)
    return file_path, None, 200

def run_model_on_video(video_path):
    if not video_model or not cnn_model:
        return "Model is not loaded."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Failed to open video file."

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < SEQUENCE_LENGTH:
        cap.release()
        return f"Video must have at least {SEQUENCE_LENGTH} frames."

    frame_indices = np.linspace(0, frame_count - 1, SEQUENCE_LENGTH, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "Failed to read frame during sampling."
        frame = cv2.resize(frame, IMG_SIZE) / 255.0
        frames.append(frame)

    cap.release()
    frames = np.array(frames).reshape(-1, *IMG_SIZE, 3)

    try:
        features = cnn_model.predict(frames, batch_size=16, verbose=0)
        features = features.reshape(1, SEQUENCE_LENGTH, features.shape[-1])
        prediction = video_model.predict(features)
        predicted_class = np.argmax(prediction)
    except Exception as e:
        return f"Prediction failed: {str(e)}"

    return CLASS_MAP.get(predicted_class, "Unknown")
