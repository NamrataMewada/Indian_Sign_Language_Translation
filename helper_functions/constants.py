# helpers/constants.py

import os
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque

# ===== Constants =====
IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 20
CLASS_MAP = {0: 'Accident', 1: 'Call', 2: 'Doctor', 3: 'Help', 4: 'Hot', 5: 'Lose', 6: 'Pain', 7: 'Thief'}

# ===== File Paths =====
VIDEO_FOLDER = "static/upload_media/videos"
AUDIO_FOLDER = "static/upload_media/recorded_speech"
EMERGENCY_SIGNS_PATH = "static/emergency_words_gif"
PROCESSED_AUDIO_TRANSLATION = "static/translated_audio"

for folder in [VIDEO_FOLDER, AUDIO_FOLDER, PROCESSED_AUDIO_TRANSLATION]:
    os.makedirs(folder, exist_ok=True)

# ===== Emergency Sign Model =====
try:
    video_model = tf.keras.models.load_model("ESL_gesture_model.h5")
    # base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    cnn_out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    cnn_model = tf.keras.Model(inputs=base_model.input, outputs=cnn_out)
    print("\n All models for video loaded\n")
except Exception as e:
    print(f"Error loading the model: {e}")
    video_model, cnn_model = None, None

# ===== Emergency Words Preload =====
emergency_words = {f.split('.')[0].lower(): f for f in os.listdir(EMERGENCY_SIGNS_PATH)}


# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# # ================== Real-Time Constants ==================
prev_prediction = ""
hold_counter = 0
hold_threshold = 15
last_hand_time = time.time()
space_added = False

# Better buffer using deque
prediction_buffer = deque(maxlen=15)  # Buffer size for smoothing
confidence_threshold = 10  # Min occurrences in buffer to consider stable

# Sentence buffers
current_sentence = ""
final_sentence = ""
final_sentences_history = []

# Constants for duplicate letter control
last_letter_added = ""
last_added_time = 0
letter_repeat_cooldown = 1.0  # in seconds

