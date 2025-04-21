# helper_functions/db_utils.py

from sqlalchemy.exc import SQLAlchemyError
from model import GestureTranslation, MediaDataset
from db_init import db

def save_translation_to_db(user_id, sentence):
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
        db.session.close()
        print(f"[DB ERROR] {str(e)}")

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
        return result.file_path.replace("\\", "/") if result else None

    except SQLAlchemyError as e:
        print(f"[ORM ERROR] {str(e)}")
        return None
