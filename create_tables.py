# create_tables.py

from app import app
from db_init import db

with app.app_context():
    db.create_all()
    print("\nTables created successfully!\n")
