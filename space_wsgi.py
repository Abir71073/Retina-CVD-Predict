"""Seed the first Space account before accepting HTTP requests."""
import os

from app import app
from auth import create_user, db

with app.app_context():
    if not db().execute("SELECT 1 FROM users LIMIT 1").fetchone():
        username = os.environ.get("RETINA_OWNER_USERNAME")
        password = os.environ.get("RETINA_OWNER_PASSWORD")
        if not username or not password:
            raise RuntimeError("Set RETINA_OWNER_USERNAME and RETINA_OWNER_PASSWORD in Space Settings.")
        create_user(username, password)
