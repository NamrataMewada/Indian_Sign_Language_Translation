from flask_bcrypt import Bcrypt
from flask_session import Session
from config import Config
from model import db, GestureTranslation, MediaDataset, UserUpload, User, TextSignMapping, Feedback
from flask import Flask, render_template, request, jsonify, Response, session, copy_current_request_context
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time
import pickle
from collections import deque, Counter
import cv2
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename
from flask_jwt_extended import create_access_token, JWTManager
from gtts import gTTS
import speech_recognition as sr
from datetime import datetime

# from helper_functions.constants import *
# from helper_functions.db_utils import fetch_media_from_db
# from helper_functions.video_processing import *

app = Flask(__name__)
app.config.from_object(Config)

# Extensions
db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
Session(app)

# ================== Constants ==================
IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 20  # Number of frames required for prediction
CLASS_MAP = {0: 'Accident', 1: 'Call', 2: 'Doctor', 3: 'Help', 4: 'Hot', 5: 'Lose', 6: 'Pain', 7: 'Thief'}

# ================== File Paths ==================
VIDEO_FOLDER = "static/upload_media/videos"
AUDIO_FOLDER = "static/upload_media/recorded_speech"
EMERGENCY_SIGNS_PATH = "static/emergency_words_gif"
PROCESSED_AUDIO_TRANSLATION = "static/translated_audio"

# ================== Creating the folders for uploads ==================
for folder in [VIDEO_FOLDER, AUDIO_FOLDER,PROCESSED_AUDIO_TRANSLATION]:
    os.makedirs(folder, exist_ok=True)


# ==================Preload available emergency words ==================
emergency_words = {f.split('.')[0].lower(): f for f in os.listdir(EMERGENCY_SIGNS_PATH)}


# ================== Emergency words model loading ==================
try:
    video_model = tf.keras.models.load_model("Gesture_cnn_lstm_model.h5")
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    cnn_out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    cnn_model = tf.keras.Model(inputs=base_model.input, outputs=cnn_out)
except Exception as e:
    print(f"Error loading the model: {e}")
    video_model, cnn_model = None, None

# ================== Initialize Mediapipe Hands ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils


# ================== Alphabets and Numbers Model Loading ==================  
try:
    with open('MODEL_LATEST_TUNING_augment_islrtc.p','rb') as f:
        model_dict = pickle.load(f)
        alpha_model = model_dict['model']     
except Exception as e:
    print(f"Error loading the model: {e}")
    alpha_model = None
    
# ================== Real-Time Constants ==================
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

# ================== Hand Landmark Extraction ==================
def extract_hand_landmarks(img):
    """Extract hand landmarks from an image/frame and normalize coordinates."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []
    hand_landmarks_list = results.multi_hand_landmarks or []

    if len(hand_landmarks_list) == 1:
        hand_landmarks_list.append(None) 

    for hand_landmarks in hand_landmarks_list:
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

# ================== Saving the translated sentence in database ==================
def save_translation_to_db(user_id, sentence):
    print(f"\nSentence entered into save_db function with \nuser_id: {user_id} \nSentence: {sentence}\n")
    try:
        existing = GestureTranslation.query.filter_by(user_id=user_id, translated_sentence=sentence).first()
        if not existing:
            new_translation = GestureTranslation(user_id=user_id, translated_sentence=sentence)
            db.session.add(new_translation)
            db.session.commit()
            print(f"[DB] Saved: {sentence}")
        else:
            print(f"[DB] Duplicate not saved: {sentence}")
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"[DB ERROR] Failed to save sentence: {str(e)}")

# ================== Real-Time Frame Generation ==================
def generate_frames(user_id):
    global current_sentence, final_sentence, prev_prediction
    global hold_counter, last_hand_time, space_added, prediction_buffer
    global final_sentences_history, last_letter_added, last_added_time
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while cap and cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Extract hand landmarks
            data_np = extract_hand_landmarks(frame)

            gesture_text = "Waiting for gesture..."
            space_text = ""
            display_current = current_sentence
            display_final = final_sentence

            if data_np is not None:
                last_hand_time = time.time()
                space_added = False

                # Prediction and buffering
                prediction = alpha_model.predict(data_np)[0]
                prediction_buffer.append(prediction)

                # Get most common prediction and its frequency
                most_common_pred = Counter(prediction_buffer).most_common(1)[0]
                stable_prediction, freq = most_common_pred

                # Confidence filtering
                if freq >= confidence_threshold:
                    if stable_prediction == prev_prediction:
                        hold_counter += 1
                    else:
                        hold_counter = 1
                        prev_prediction = stable_prediction

                    # Confirm gesture
                    if hold_counter >= hold_threshold:
                        current_time = time.time()
                        
                        # Check if this is a new letter or enough time has passed since same letter was added
                        if stable_prediction != last_letter_added or (current_time - last_added_time) > letter_repeat_cooldown:
                            current_sentence += stable_prediction
                            print(f"[Letter Added] Current Sentence: {current_sentence}")
                            
                            last_letter_added = stable_prediction
                            last_added_time = current_time
                            
                        # Reset the tracking
                        hold_counter = 0
                        prediction_buffer.clear()

                    gesture_text = f"Gesture: {stable_prediction}"
            
            else:
                # No hand detected
                elapsed = time.time() - last_hand_time
                
                if elapsed > 3 and not space_added and not current_sentence.endswith(" "):
                    current_sentence += " "
                    space_text = "Space Added"
                    space_added = True
                    print(f"[Space] Current Sentence: {current_sentence}")
                    
                    # Reset duplicate letter tracking 

                elif elapsed > 7:
                    if current_sentence.strip():
                        final_sentence = current_sentence.strip()
                        final_sentences_history.append(final_sentence)
                        
                        # Store the final sentence in the db
                        save_translation_to_db(user_id, final_sentence)
                        print(f"[Final] Final Sentence: {final_sentence}")
                        print(f"All sentence list: ", final_sentences_history)

                        # Reset for next sentence
                        current_sentence = ""
                        prediction_buffer.clear()
                        hold_counter = 0
                        space_added = False
                        last_letter_added = ""
                        time.sleep(2)

                    last_hand_time = time.time()

            # Draw landmarks
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display text overlays
            cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(frame, space_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"Current: {display_current}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"Final: {display_final}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
    except GeneratorExit:
        print("[Stream] Client disconnected")
    except Exception as e:
        print(f"[Stream Error] {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Stream] Camera released properly")

# ================== Upload file handler ==================
def get_uploaded_file(file_key):
    if file_key not in request.files:
        return None, jsonify({"error": f"No {file_key} uploaded"}), 400
    
    file = request.files[file_key]
    if file.filename == '':
        return None, jsonify({"error": f"No selected {file_key} file"}), 400
    
    filename = secure_filename(file.filename)
    
    if file_key == "video":
        file_path = os.path.join(VIDEO_FOLDER, filename)
    else:
        return "Plaese upload valid file"
    
    file.save(file_path)
    
    return file_path, None, 200 

# ================== Upload file handler ==================
def run_model_on_video(video_path):
    """Process uploaded video, extract frames, and run emergency sign prediction."""
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

    if len(frames) != SEQUENCE_LENGTH:
        return "Could not extract sufficient frames from the video."

    frames = np.array(frames).reshape(-1, *IMG_SIZE, 3)

    try:
        features = cnn_model.predict(frames, batch_size=16, verbose=0)
        features = features.reshape(1, SEQUENCE_LENGTH, features.shape[-1])
        prediction = video_model.predict(features)
        predicted_class = np.argmax(prediction)
    except Exception as e:
        return f"Prediction failed: {str(e)}"

    return CLASS_MAP.get(predicted_class, "Unknown")

# ================== Fetching the media from the database ==================
def fetch_media_from_db(media_type, category=None, char=None):
    try:
        if not all([media_type, category, char]):
            print("[ORM] Missing required query parameters")
            return None

        query = MediaDataset.query.filter_by(media_type=media_type, category=category)

        if media_type == "image":
            query = query.filter(MediaDataset.file_path.ilike(f"%/{char.upper()}%"))
        elif media_type == "video":
            query = query.filter(MediaDataset.file_path.ilike(f"%{char.lower()}%"))
        else:
            print("[ORM] Unsupported media type")
            return None

        result = query.first()

        if result:
            return result.file_path.replace("\\", "/")
        else:
            print("[ORM] No matching media found")
            return None

    except SQLAlchemyError as e:
        print(f"[ORM ERROR] {str(e)}")
        return None
    
    
# ======================================================================== #
# ================== Main routing ========================================

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# About Page route
@app.route('/about')
def about():
    return render_template('about.html')

# Service Page route
@app.route('/services')
def services():
    return render_template('services.html')

# Feedback Page route 
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# ================== Processing the video API ==================
@app.route('/process_video', methods=['POST'])
def process_video():
    user_id = session.get("user_id")
    video_path, error_response, status = get_uploaded_file("video")

    if error_response:
        return error_response, status

    result_text = run_model_on_video(video_path)
    processed_status = bool(result_text)

    try:
        # Check if file already exists for this user
        existing_upload = UserUpload.query.filter_by(user_id=user_id, file_path=video_path).first()

        if existing_upload:
            return jsonify({
                "warning": "You have already uploaded this file.",
                "result": result_text
            })

        # Insert new record using SQLAlchemy ORM
        new_upload = UserUpload(
            user_id=user_id,
            media_type="video",
            file_path=video_path,
            processed_status=processed_status,
            prediction_result=result_text
        )
        db.session.add(new_upload)
        db.session.commit()

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({"result": result_text})
  
# ================== Signup Route ==================
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    raw_password = data.get("password")

    # Input validation
    if not username or not email or not raw_password:
        return jsonify({"error": "All fields are required"}), 400

    # Hash the password
    password = bcrypt.generate_password_hash(raw_password).decode("utf-8")

    try:
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"error": "Email already exists"}), 400

        # Create new user object
        new_user = User(username=username, email=email, password=password)

        # Add to DB using ORM
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User registered successfully!"}), 201

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400
    
# ================== Login Route ==================
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    # Validate input
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        # Fetch user by email
        user = User.query.filter_by(email=email).first()

        # Check password
        if user and bcrypt.check_password_hash(user.password, password):
            access_token = create_access_token(identity=user.user_id)
            session["user_id"] = user.user_id  # Set session

            return jsonify({
                "token": access_token,
                "username": user.username,
                "email": user.email,
                "message": "Login successful!"
            })

        return jsonify({"error": "Invalid credentials"}), 401

    except SQLAlchemyError as e:
        return jsonify({"error": str(e)}), 500   
    
# ================== Update the profile ==================  
@app.route("/update_profile", methods=["POST"])
def update_profile():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email:
        return jsonify({"error": "Username and email are required"}), 400

    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Check if email is already taken by another user
        existing_user = User.query.filter(User.email == email, User.user_id != user_id).first()
        if existing_user:
            return jsonify({"error": "Email already in use by another account"}), 409

        # Update fields
        user.username = username
        user.email = email
        if password:
            user.password = bcrypt.generate_password_hash(password).decode("utf-8")

        db.session.commit()
        return jsonify({"message": "Profile updated successfully!"})

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
# ================== Text to Sign processing ==================
@app.route('/process_text', methods=['POST'])
def process_text():
    user_id = session.get("user_id")
    data = request.get_json() or {}
    text = data.get("text", "").strip().lower()
    input_type = data.get("input_type", "text")
    audio_path = data.get("audio_path", None)

    # Clean text input
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    if not text or text == "undefined":
        return jsonify({"error": "No text provided"}), 400

    words = text.split()
    result = []

    for word in words:
        if word in emergency_words:
            gif_path = fetch_media_from_db("video", category="Emergency Sign", char=word)
            if gif_path:
                result.append({
                    "type": "gif",
                    "path": gif_path,
                    "label": word.capitalize()
                })
        else:
            word_data = {
                "type": "images",
                "paths": [],
                "label": word.capitalize()
            }
            for char in word:
                if char.isalnum():
                    if char == "0":
                        char = "o"
                    img_path = fetch_media_from_db(
                        "image",
                        category="Alphabet" if char.isalpha() else "Number",
                        char=char
                    )
                    if img_path:
                        word_data["paths"].append(img_path)
            result.append(word_data)

    status = "processed" if result else "failed"

    try:
        existing_entry = TextSignMapping.query.filter_by(user_id=user_id, sentence=text).first()
        if existing_entry:
            return jsonify({
                "status": status,
                "result": result,
                "warning": "You have translated the same sentence again"
            })

        new_entry = TextSignMapping(
            user_id=user_id,
            input_type=input_type,
            audio_path=audio_path,
            sentence=text,
            list_of_words=words,
            status=status
        )

        db.session.add(new_entry)
        db.session.commit()

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({"result": result})

# ================== Speech to text handling ==================
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)

        filename = f"speech_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        file_path = os.path.join(AUDIO_FOLDER, filename)

        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())

        text = recognizer.recognize_google(audio, language="en-IN")

        return jsonify({
            "text": text,
            "input_type": "speech",
            "audio_path": file_path.replace("\\", "/")
        })

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio"}), 400

    except sr.RequestError:
        return jsonify({"error": "Speech Recognition Service error."}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ================== Text to speech conversion ==================
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json() or {}
    text = data.get("text", "").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    audio_filename = f"{text}.mp3"
    audio_path = os.path.join(PROCESSED_AUDIO_TRANSLATION, audio_filename)

    try:
        if os.path.exists(audio_path):
            return jsonify({"audio_url": audio_path.replace("\\", "/")})

        tts = gTTS(text=text, lang='en')
        tts.save(audio_path)

        return jsonify({"audio_url": audio_path.replace("\\", "/")})

    except Exception as e:
        return jsonify({"error": f"Text-to-speech failed: {str(e)}"}), 500

# ================== Feedback submission ==================
@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    name = data.get("name")
    email = data.get("email")
    category = data.get("category")
    rating = data.get("rating")
    message = data.get("message")

    # Basic validation
    if not name or not message or not rating:
        return jsonify({"error": "Name, rating, and message are required"}), 400

    try:
        feedback = Feedback(
            user_id=user_id,
            name=name,
            email=email,
            category=category,
            rating=int(rating),
            message=message
        )
        db.session.add(feedback)
        db.session.commit()

        return jsonify({"message": "Feedback submitted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

# ================== Video Response from camera to frontend ==================          
@app.route('/video_feed')
def video_feed():
    user_id = session.get("user_id")
    @copy_current_request_context
    def wrapped_generate():
        return generate_frames(user_id)

    return Response(wrapped_generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed')
# def video_feed():
#     user_id = session.get("user_id")

#     @copy_current_request_context
#     def wrapped_generate():
#         try:
#             return generate_frames(user_id)
#         except Exception as e:
#             print(f"[ERROR] in video_feed: {str(e)}")
#             return iter([])  # Empty response stream

#     return Response(wrapped_generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============ Sending the translated sentence in real time to frontend for display ==================
@app.route('/get_translation')
def get_translation():
    return jsonify({
        "gesture": prev_prediction,
        "current_text": current_sentence,
        "final_sentence": final_sentence})

# ================== Sending all the senetnces generated by user to store and display ==================
@app.route('/get_all_sentences')
def get_all_sentences():
    return jsonify({
        "all_sentences": final_sentences_history
    })  
    
if __name__ == "__main__":
    app.run(debug=True)
