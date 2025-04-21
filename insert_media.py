# insert_media.py

import os
from model import MediaDataset
from db_init import db
from app import app

alphabet_folder = "static/alphabets_numbers"
gif_folder = "static/emergency_words_gif"

def insert_alphabets_and_gifs():
    with app.app_context():
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            file_path = None
            for ext in ["png", "PNG", "jpg", "jpeg"]:
                temp_path = os.path.join(alphabet_folder, f"{letter if letter != '0' else 'O'}.{ext}")
                temp_path = os.path.normpath(temp_path).replace("\\", "/")
                if os.path.exists(temp_path):
                    file_path = temp_path
                    break

            if file_path:
                category = "Alphabet" if letter.isalpha() else "Number"
                if not MediaDataset.query.filter_by(file_path=file_path).first():
                    entry = MediaDataset(
                        media_type="image",
                        label = letter,
                        file_path=file_path,
                        category=category,
                        description=f"Sign language for '{letter}'"
                    )
                    db.session.add(entry)

        for gesture in ["call", "doctor", "help", "hot", "lose", "pain", "accident", "thief"]:
            file_path = os.path.join(gif_folder, f"{gesture}.gif")
            file_path = os.path.normpath(file_path).replace("\\", "/")
            if os.path.exists(file_path):
                if not MediaDataset.query.filter_by(file_path=file_path).first():
                    entry = MediaDataset(
                        media_type="video",
                        label = gesture,
                        file_path=file_path,
                        category="Emergency Sign",
                        description=f"Emergency gesture for '{gesture}'"
                    )
                    db.session.add(entry)

        db.session.commit()
        print("\nAll media inserted successfully!\n")

if __name__ == "__main__":
    insert_alphabets_and_gifs()
