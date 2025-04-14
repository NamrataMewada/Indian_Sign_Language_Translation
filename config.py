# import os

# DB_USERNAME = 'postgres'
# DB_PASSWORD = 'Namrata'
# DB_HOST = 'localhost'
# DB_PORT = '5432'
# DB_NAME = 'Sign_Language_Translator'

# SQLALCHEMY_DATABASE_URI = (
#     f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# )


import os

class Config:
    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:Namrata@localhost:5432/Sign_Language_Translator"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = "secret_key"
    SESSION_TYPE = "filesystem"
