import os
from gtts import gTTS
from db_init import db
from config import Config
from datetime import datetime
import speech_recognition as sr
from flask_bcrypt import Bcrypt
from flask_session import Session
from sqlalchemy.exc import SQLAlchemyError
from model import  UserUpload, User, TextSignMapping, Feedback
from flask_jwt_extended import create_access_token, JWTManager
from flask import Flask, render_template, request, jsonify, Response, session, copy_current_request_context

from helper_functions.constants import *
from helper_functions.db_utils import *
from helper_functions.video_processing import *
from helper_functions import realtime_utils
from helper_functions.realtime_utils import generate_frames, load_model


app = Flask(__name__)
app.config.from_object(Config)

# Extensions
db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
Session(app)

load_model()
    
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

        # Insert new record using SQLAlchemy 
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

# ============ Sending the translated sentence in real time to frontend for display ==================
@app.route('/get_translation')
def get_translation():
    # print(f"\n Sentence in get translation code of backend with \n Prev_prediction: {prev_prediction} \ncurrent_text: {current_sentence} \n Final Sentence: {final_sentence} \n")
    return jsonify({
        "gesture": realtime_utils.prev_prediction,
        "current_text": realtime_utils.current_sentence,
        "final_text": realtime_utils.final_sentence})

# ================== Sending all the senetnces generated by user to store and display ==================
@app.route('/get_all_sentences')
def get_all_sentences():
    return jsonify({
        "all_sentences": final_sentences_history
    })  
    
if __name__ == "__main__":
    app.run(debug=True)
