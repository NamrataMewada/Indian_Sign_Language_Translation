from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MediaDataset(db.Model):
    __tablename__ = 'media_dataset'
    id = db.Column(db.Integer, primary_key=True)
    media_type = db.Column(db.String(10), nullable=False)
    file_path = db.Column(db.Text, unique=True, nullable=False)
    category = db.Column(db.String(100))
    description = db.Column(db.Text)

class UserUpload(db.Model):
    __tablename__ = 'user_uploads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete="CASCADE"), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)
    file_path = db.Column(db.Text, nullable=False)
    processed_status = db.Column(db.Boolean, default=False)
    prediction_result = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id', 'file_path', name='unique_user_file'),)

class GestureTranslation(db.Model):
    __tablename__ = 'gesture_translations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete="CASCADE"), nullable=False)
    translated_sentence = db.Column(db.Text, nullable=False)
    translated_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id', 'translated_sentence', name='unique_translation'),)

class TextSignMapping(db.Model):
    __tablename__ = 'text_sign_mapping'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete="CASCADE"), nullable=False)
    input_type = db.Column(db.Text, nullable=False)
    audio_path = db.Column(db.Text)
    sentence = db.Column(db.Text, nullable=False)
    list_of_words = db.Column(db.ARRAY(db.Text), nullable=False)
    status = db.Column(db.Text, default='pending')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id', 'sentence', name='unique_sentence'),)

    @property
    def word_count(self):
        return len(self.list_of_words) if self.list_of_words else 0

class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id', ondelete="CASCADE"))
    name = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, nullable=False)
    category = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)
    message = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
