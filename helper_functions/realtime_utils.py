# helper_functions/realtime_utils.py

import time
import cv2
from collections import Counter
import pickle
import os 
from .constants import *
from .preprocessing import extract_hand_landmarks
from .db_utils import save_translation_to_db


alpha_model = None

def load_model():
    global alpha_model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Random_forest_model_tuned.p') 
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            alpha_model = model_dict['model']
        print("[Model] Loaded successfully.")
    except Exception as e:
        print(f"[Model Error] Failed to load model: {e}")

# ================== Real-Time Frame Generation ==================
def generate_frames(user_id):
    from app import app
    
    global current_sentence, final_sentence, prev_prediction
    global hold_counter, last_hand_time, space_added, prediction_buffer
    global final_sentences_history, last_letter_added, last_added_time
    
    if not alpha_model:
        print("Alpha model not loaded properly.")
        return

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
                        
                        with app.app_context():
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
