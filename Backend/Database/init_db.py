import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

db = sqlite3.connect(DB_PATH)
cur = db.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    password TEXT,
    name TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    result TEXT,
    image_path TEXT,
    timestamp TEXT
)
""")

cur.execute("""
INSERT OR IGNORE INTO users (user_id, password, name)
VALUES ('user101', 'pass123', 'Test User')
""")
cur.execute("""
INSERT OR IGNORE INTO users (user_id, password, name)
VALUES ('admin', 'admin123', 'Administrator')
""")

db.commit()
db.close()

print("✅ Database initialized successfully at:", DB_PATH)
