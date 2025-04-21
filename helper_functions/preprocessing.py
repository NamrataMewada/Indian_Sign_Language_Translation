# helper_functions/preprocessing.py

import cv2
import numpy as np
from .constants import hands

def extract_hand_landmarks(img):
    """Extract hand landmarks from an image/frame and normalize coordinates."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []
    hand_landmarks_list = results.multi_hand_landmarks or []

    while len(hand_landmarks_list) < 2:
        hand_landmarks_list.append(None) 

    for hand_landmarks in hand_landmarks_list[:2]:
        x_vals, y_vals = [], []
        if hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_vals.append(landmark.x)
                y_vals.append(landmark.y)

            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)

            for landmark in hand_landmarks.landmark:
                norm_x = (landmark.x - min_x) / (max_x - min_x + 1e-6)
                norm_y = (landmark.y - min_y) / (max_y - min_y + 1e-6)
                data_aux.extend([norm_x, norm_y])
        else:
            data_aux.extend([0] * 42)

    return np.array([data_aux]) if len(data_aux) == 84 else None
