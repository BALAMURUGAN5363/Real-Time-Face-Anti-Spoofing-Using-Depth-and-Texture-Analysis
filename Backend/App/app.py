from flask import Flask, render_template, request, redirect, session
import sqlite3
import cv2
import base64
import numpy as np
from datetime import datetime
from realtime.anti_spoof import predict_spoof
import os

app = Flask(__name__)
app.secret_key = "secure_exam_key"

# Admin allowlist: set the usernames that are allowed to access admin
admin_allowlist = {"admin"}  # TODO: add your admin usernames here, e.g., {"admin", "admin1"}

# =========================
# DATABASE CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

def get_db():
    return sqlite3.connect(DB_PATH)

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# LOGIN PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uid = request.form["user_id"]
        pwd = request.form["password"]

        db = get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT * FROM users WHERE user_id=? AND password=?",
            (uid, pwd)
        )
        user = cur.fetchone()
        db.close()

        if user:
            session["user"] = uid
            return redirect("/camera")

        return "❌ Invalid Credentials"

    return render_template("login.html")

# =========================
# ADMIN LOGIN PAGE
# =========================
@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        uid = request.form["user_id"]
        pwd = request.form["password"]

        db = get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT * FROM users WHERE user_id=? AND password=?",
            (uid, pwd)
        )
        user = cur.fetchone()
        db.close()

        if user and uid in admin_allowlist:
            session["admin"] = uid
            return redirect("/admin")

        return "❌ Invalid Admin Credentials"

    return render_template("admin_login.html")

# =========================
# CAMERA PAGE
# =========================
@app.route("/camera")
def camera():
    if "user" not in session:
        return redirect("/")
    return render_template("camera.html")

# =========================
# VERIFY SPOOF
# =========================
@app.route("/verify", methods=["POST"])
def verify():
    uid = session.get("user")
    if not uid:
        return redirect("/")

    # Decode image from browser
    img_b64 = request.form["image"].split(",")[1]
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(img_b64), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return "❌ Invalid image"

    # Detect face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "❌ No face detected. Please retry."

    # Take first detected face
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    # =========================
    # 🔥 SPOOF PREDICTION (SIMPLE & CORRECT)
    # =========================
    score = predict_spoof(face)
    print("Spoof score:", score)

    timestamp = int(datetime.now().timestamp())

    # =========================
    # THRESHOLD LOGIC (YOUR REQUEST)
    # =========================
    if score > 0.50:
        result = "REAL"
        folder = "real"
        page = "exam.html"
    else:
        result = "SPOOF"
        folder = "spoof"
        page = "blocked.html"

    # Save image
    save_dir = os.path.join(BASE_DIR, "static", "captures", folder)
    os.makedirs(save_dir, exist_ok=True)

    img_path = os.path.join(save_dir, f"{uid}_{timestamp}.jpg")
    cv2.imwrite(img_path, face)

    # Store log
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO login_logs (user_id, result, image_path, timestamp) VALUES (?, ?, ?, ?)",
        (uid, result, img_path, datetime.now())
    )
    db.commit()
    db.close()

    return render_template(page)

# =========================
# ADMIN DASHBOARD
# =========================
@app.route("/admin")
def admin():
    if "admin" not in session:
        return redirect("/admin-login")
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT user_id, result, image_path, timestamp
        FROM login_logs
        ORDER BY timestamp DESC
    """)
    logs = cur.fetchall()
    db.close()

    # Convert absolute paths → /static paths
    processed_logs = []
    for log in logs:
        user_id, result, image_path, timestamp = log
        image_url = image_path.replace(BASE_DIR, "").replace("\\", "/")
        processed_logs.append((user_id, result, image_url, timestamp))

    return render_template("admin.html", logs=processed_logs)

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)
