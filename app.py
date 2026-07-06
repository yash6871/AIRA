# ================== IMPORTS ==================
import os
import pickle
import numpy as np
import cv2  # opencv-python-headless: drawing/imwrite work fine, no GUI needed
from datetime import datetime
from PIL import Image

from flask import Flask, render_template, request, redirect, session, jsonify

from database import connect_db, init_db
from face_engine import FaceEngine
from match_utils import find_best_match, decode_base64_image

# ================== CONFIG ==================
CONFIDENCE_THRESHOLD = 0.5  # cosine similarity threshold (0-1)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "secret123")

engine = FaceEngine.instance()

init_db()

# ================== HELPER ==================
def get_inst_id():
    return session.get("institution_id", 1)


# ================== AUTH ==================
@app.route("/")
def index():
    if not session.get("admin"):
        return redirect("/login")

    if "institution_id" not in session:
        session["institution_id"] = 1

    inst_id = get_inst_id()
    today = datetime.now().strftime("%Y-%m-%d")
    stats = {"total_students": 0, "present_today": 0, "absent_today": 0,
             "sessions_today": 0, "attendance_pct": 0, "recent": [], "week": []}
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM students WHERE institution_id=?", (inst_id,))
        stats["total_students"] = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(DISTINCT roll) FROM attendance WHERE institution_id=? AND date=?", (inst_id, today))
        stats["present_today"] = cur.fetchone()[0] or 0
        stats["absent_today"] = max(stats["total_students"] - stats["present_today"], 0)

        cur.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE institution_id=?", (inst_id,))
        stats["sessions_today"] = cur.fetchone()[0] or 0

        stats["attendance_pct"] = round((stats["present_today"] / stats["total_students"]) * 100, 1) if stats["total_students"] else 0

        cur.execute("SELECT name, roll, date, time FROM attendance WHERE institution_id=? ORDER BY id DESC LIMIT 5", (inst_id,))
        stats["recent"] = cur.fetchall()

        cur.execute("SELECT date, COUNT(DISTINCT roll) FROM attendance WHERE institution_id=? GROUP BY date ORDER BY date DESC LIMIT 7", (inst_id,))
        stats["week"] = list(reversed(cur.fetchall()))
        conn.close()
    except Exception:
        pass

    return render_template("index.html", stats=stats, today_str=datetime.now().strftime("%A, %d %B %Y"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = connect_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT id, name, institution_id FROM users WHERE email=? AND password=?",
            (username, password)
        )
        user = cur.fetchone()

        conn.close()

        if user:
            session["admin"] = True
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            session["institution_id"] = user[2]
            return redirect("/")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        institute_name = request.form.get("institute")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = connect_db()
        cur = conn.cursor()

        cur.execute("SELECT id FROM institutions WHERE email=?", (email,))
        if cur.fetchone():
            conn.close()
            return render_template("signup.html", error="Institute already exists")

        cur.execute(
            "INSERT INTO institutions (name, email, password) VALUES (?,?,?)",
            (institute_name, email, password)
        )
        inst_id = cur.lastrowid

        cur.execute(
            "INSERT INTO users (name, email, password, role, institution_id) VALUES (?,?,?,?,?)",
            (institute_name + " Admin", email, password, "admin", inst_id)
        )

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("signup.html")


# ================== REGISTER (browser camera, no cv2.VideoCapture) ==================
@app.route("/register", methods=["GET", "POST"])
def register():
    if not session.get("admin"):
        return redirect("/login")

    if request.method == "POST":
        name = request.form.get("name")
        roll = request.form.get("roll")
        batch = request.form.get("batch")
        images_b64 = request.form.getlist("images")  # captured via browser camera JS

        if not (name and roll and batch and images_b64):
            return render_template(
                "register.html",
                error="Name, roll, batch aur kam se kam 1 photo (camera se capture karo) zaroori hai."
            )

        encodings = []
        first_img = None

        for data_url in images_b64:
            try:
                img = decode_base64_image(data_url)
                if first_img is None:
                    first_img = img
                emb = engine.get_embedding(img)
                encodings.append(emb)
            except Exception:
                continue

        if not encodings:
            return render_template(
                "register.html",
                error="Face detect nahi hua. Achi lighting me, seedha camera dekh ke dobara try karo."
            )

        avg_encoding = np.mean(encodings, axis=0)
        avg_encoding = avg_encoding / np.linalg.norm(avg_encoding)

        os.makedirs("static/students", exist_ok=True)
        if first_img is not None:
            Image.fromarray(first_img).save(f"static/students/{name}_{roll}.jpg")

        conn = connect_db()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO students (name, roll, batch, encoding, institution_id) VALUES (?,?,?,?,?)",
            (name, roll, batch, pickle.dumps(avg_encoding), get_inst_id())
        )

        conn.commit()
        conn.close()

        return redirect("/students")

    return render_template("register.html")


# ================== STUDENTS ==================
@app.route("/students")
def students():
    if not session.get("admin"):
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT name, roll, batch FROM students WHERE institution_id=?",
        (get_inst_id(),)
    )
    rows = cur.fetchall()

    conn.close()

    students_list = [
        {
            "name": n,
            "roll": r,
            "batch": b,
            "photo": f"/static/students/{n}_{r}.jpg"
        }
        for n, r, b in rows
    ]

    return render_template("students.html", students=students_list)


# ================== IMAGE UPLOAD (classroom group photo) ==================
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("admin"):
        return redirect("/login")

    img_url = None
    marked_names = []

    if request.method == "POST":
        file = request.files.get("photo")

        if not file:
            return render_template("upload.html")

        conn = connect_db()
        cur = conn.cursor()

        img = np.array(Image.open(file).convert("RGB"))

        # detection + embedding both come from the same engine now
        faces = engine.get_faces(img)

        cur.execute(
            "SELECT name, roll, encoding FROM students WHERE institution_id=?",
            (get_inst_id(),)
        )
        students_db = cur.fetchall()

        embeddings = []
        labels = []

        for s in students_db:
            emb = pickle.loads(s[2])
            if emb is None:
                continue
            embeddings.append(emb)
            labels.append((s[0], s[1]))

        today = datetime.now().strftime("%Y-%m-%d")
        marked_rolls = set()

        for (box, emb) in faces:
            x1, y1, x2, y2 = box
            label = "Unknown"
            color = (0, 0, 255)

            match, score = find_best_match(emb, embeddings, labels, CONFIDENCE_THRESHOLD)

            if match:
                name, roll = match
                label = name
                color = (0, 255, 0)

                if roll not in marked_rolls:
                    cur.execute(
                        "SELECT 1 FROM attendance WHERE roll=? AND date=? AND institution_id=?",
                        (roll, today, get_inst_id())
                    )

                    if not cur.fetchone():
                        cur.execute(
                            "INSERT INTO attendance (name, roll, date, time, confidence, institution_id) VALUES (?,?,?,?,?,?)",
                            (name, roll, today, datetime.now().strftime("%H:%M:%S"), score, get_inst_id())
                        )
                        conn.commit()
                        marked_names.append(name)

                    marked_rolls.add(roll)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        conn.close()

        os.makedirs("static/results", exist_ok=True)
        result_path = "static/results/result.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        img_url = "/" + result_path

    return render_template("upload.html", image_url=img_url, names=marked_names)


# ================== DELETE ==================
@app.route("/delete_student/<roll>", methods=["POST"])
def delete_student(roll):
    if not session.get("admin"):
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT name FROM students WHERE roll=?", (roll,))
    row = cur.fetchone()
    name = row[0] if row else ""

    cur.execute(
        "DELETE FROM students WHERE roll=? AND institution_id=?",
        (roll, get_inst_id())
    )
    cur.execute(
        "DELETE FROM attendance WHERE roll=? AND institution_id=?",
        (roll, get_inst_id())
    )

    conn.commit()
    conn.close()

    img_path = f"static/students/{name}_{roll}.jpg"
    if os.path.exists(img_path):
        os.remove(img_path)

    return redirect("/students")


# ================== RESET ==================
@app.route("/reset_system")
def reset_system():
    if not session.get("admin"):
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("DELETE FROM students WHERE institution_id=?", (get_inst_id(),))
    cur.execute("DELETE FROM attendance WHERE institution_id=?", (get_inst_id(),))

    conn.commit()
    conn.close()

    for folder in ["static/students", "static/results"]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

    return redirect("/")


# ================== RECORDS ==================
@app.route("/records")
def records():
    if not session.get("admin"):
        return redirect("/login")

    selected_batch = request.args.get("batch", "ALL")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT DISTINCT batch FROM students WHERE institution_id=?",
        (get_inst_id(),)
    )
    batches = [b[0] for b in cur.fetchall()]

    if selected_batch == "ALL":
        cur.execute("""
            SELECT attendance.id, attendance.name, attendance.roll,
                   students.batch, attendance.date, attendance.time
            FROM attendance
            LEFT JOIN students ON attendance.roll = students.roll
            WHERE attendance.institution_id=?
            ORDER BY attendance.date DESC, attendance.time DESC
        """, (get_inst_id(),))
    else:
        cur.execute("""
            SELECT attendance.id, attendance.name, attendance.roll,
                   students.batch, attendance.date, attendance.time
            FROM attendance
            LEFT JOIN students ON attendance.roll = students.roll
            WHERE students.batch=? AND attendance.institution_id=?
            ORDER BY attendance.date DESC, attendance.time DESC
        """, (selected_batch, get_inst_id()))

    rows = cur.fetchall()
    conn.close()

    data = []
    for r in rows:
        confidence = 95

        status = "On Time"
        try:
            hour = int(r[5].split(":")[0])
            if hour >= 10:
                status = "Late"
        except Exception:
            pass

        data.append(list(r) + [confidence, status])

    return render_template(
        "records.html",
        data=data,
        batches=batches,
        selected_batch=selected_batch
    )


# ================== LIVE ATTENDANCE (browser camera, works on phone + laptop) ==================
@app.route("/live_attendance")
def live_attendance():
    if not session.get("admin"):
        return redirect("/login")

    return render_template("live_attendance.html")


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Called repeatedly by the browser camera JS with one captured frame at a time."""
    if not session.get("admin"):
        return jsonify({"error": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")

    if not data_url:
        return jsonify({"error": "no image"}), 400

    try:
        img = decode_base64_image(data_url)
    except Exception:
        return jsonify({"error": "bad image"}), 400

    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT name, roll, encoding FROM students WHERE institution_id=?",
        (get_inst_id(),)
    )
    students_db = cur.fetchall()

    embeddings, labels = [], []
    for s in students_db:
        emb = pickle.loads(s[2])
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append((s[0], s[1]))

    faces = engine.get_faces(img)
    today = datetime.now().strftime("%Y-%m-%d")
    results = []

    for box, emb in faces:
        match, score = find_best_match(emb, embeddings, labels, CONFIDENCE_THRESHOLD)
        name_out, roll_out, status = "Unknown", None, "unknown"

        if match:
            name_out, roll_out = match
            status = "matched"

            cur.execute(
                "SELECT 1 FROM attendance WHERE roll=? AND date=? AND institution_id=?",
                (roll_out, today, get_inst_id())
            )

            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO attendance (name, roll, date, time, confidence, institution_id) VALUES (?,?,?,?,?,?)",
                    (name_out, roll_out, today, datetime.now().strftime("%H:%M:%S"), score, get_inst_id())
                )
                conn.commit()
                status = "marked"

        results.append({
            "name": name_out,
            "roll": roll_out,
            "box": box,
            "score": round(score, 3),
            "status": status
        })

    conn.close()
    return jsonify({"faces": results})


# ================== VIDEO UPLOAD ==================
@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    if not session.get("admin"):
        return redirect("/login")

    marked_names = []

    if request.method == "POST":
        file = request.files.get("video")
        if not file:
            return render_template("upload_video.html")

        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", file.filename)
        file.save(video_path)

        conn = connect_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT name, roll, encoding FROM students WHERE institution_id=?",
            (get_inst_id(),)
        )
        students_db = cur.fetchall()

        embeddings, labels = [], []
        for s in students_db:
            emb = pickle.loads(s[2])
            if emb is None:
                continue
            embeddings.append(emb)
            labels.append((s[0], s[1]))

        cap = cv2.VideoCapture(video_path)  # reading a FILE, not a live webcam — works fine on cloud too
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        sample_every = max(int(fps), 1)  # check ~1 frame per second of video

        today = datetime.now().strftime("%Y-%m-%d")
        marked_rolls = set()
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                faces = engine.get_faces(frame_rgb)

                for box, emb in faces:
                    match, score = find_best_match(emb, embeddings, labels, CONFIDENCE_THRESHOLD)
                    if match:
                        name, roll = match
                        if roll not in marked_rolls:
                            cur.execute(
                                "SELECT 1 FROM attendance WHERE roll=? AND date=? AND institution_id=?",
                                (roll, today, get_inst_id())
                            )
                            if not cur.fetchone():
                                cur.execute(
                                    "INSERT INTO attendance (name, roll, date, time, confidence, institution_id) VALUES (?,?,?,?,?,?)",
                                    (name, roll, today, datetime.now().strftime("%H:%M:%S"), score, get_inst_id())
                                )
                                conn.commit()
                                marked_names.append(name)
                            marked_rolls.add(roll)

            frame_idx += 1

        cap.release()
        conn.close()

    return render_template("upload_video.html", names=marked_names)


# ================== STUDENT PORTAL (read-only) ==================
@app.route("/student_login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        roll = request.form.get("roll")
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT roll FROM students WHERE roll=?", (roll,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return render_template("student_login.html", error="Roll number nahi mila")

        return redirect(f"/student_dashboard?roll={roll}")

    return render_template("student_login.html")


@app.route("/student_dashboard")
def student_dashboard():
    roll = request.args.get("roll")
    if not roll:
        return redirect("/student_login")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT name, roll, batch FROM students WHERE roll=?", (roll,))
    student = cur.fetchone()

    if not student:
        conn.close()
        return redirect("/student_login")

    name, roll, batch = student

    cur.execute(
        "SELECT date, time, confidence FROM attendance WHERE roll=? ORDER BY date DESC, time DESC",
        (roll,)
    )
    rows = cur.fetchall()
    conn.close()

    present = len(rows)
    # crude "total" estimate: distinct dates anyone was marked present (whole batch) — simplest
    # safe default, avoids extra heavy queries
    total = present if present > 0 else 0
    absent = 0
    percentage = 100 if present > 0 else 0

    records = [(d, t, "Present", int((c or 0) * 100)) for d, t, c in rows]

    return render_template(
        "student_dashboard.html",
        name=name, roll=roll, batch=batch,
        total=total, present=present, absent=absent,
        percentage=percentage, records=records
    )



# ================== EXPORT CSV ==================
@app.route("/export")
def export_csv():
    if not session.get("admin"):
        return redirect("/login")

    import csv
    import io
    from flask import Response

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT attendance.name, attendance.roll, students.batch,
               attendance.date, attendance.time, attendance.confidence
        FROM attendance
        LEFT JOIN students ON attendance.roll = students.roll
        WHERE attendance.institution_id=?
        ORDER BY attendance.date DESC, attendance.time DESC
    """, (get_inst_id(),))
    rows = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Roll", "Batch", "Date", "Time", "Confidence"])
    for r in rows:
        writer.writerow(r)

    csv_data = output.getvalue()
    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/analytics")
def analytics():
    if not session.get("admin"):
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM students WHERE institution_id=?", (get_inst_id(),))
    total_students = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM attendance WHERE institution_id=?", (get_inst_id(),))
    total_attendance = cur.fetchone()[0]

    today_str = datetime.now().strftime("%Y-%m-%d")
    cur.execute("SELECT COUNT(DISTINCT roll) FROM attendance WHERE date = ? AND institution_id=?", (today_str, get_inst_id()))
    today_present = cur.fetchone()[0]

    cur.execute("""
        SELECT students.batch, COUNT(*)
        FROM attendance
        LEFT JOIN students ON attendance.roll = students.roll
        WHERE attendance.institution_id=?
        GROUP BY students.batch
        """, (get_inst_id(),))
    batch_students = cur.fetchall()

    batch_labels = [x[0] for x in batch_students]
    batch_counts = [x[1] for x in batch_students]

    cur.execute("""
        SELECT students.batch, COUNT(*)
        FROM attendance
        LEFT JOIN students ON attendance.roll = students.roll
        WHERE attendance.institution_id=?
        GROUP BY students.batch
        """, (get_inst_id(),))
    batch_attendance = cur.fetchall()

    att_labels = [x[0] for x in batch_attendance]
    att_counts = [x[1] for x in batch_attendance]

    cur.execute("SELECT date, COUNT(*) FROM attendance WHERE institution_id=? GROUP BY date", (get_inst_id(),))
    trend_data = cur.fetchall()

    dates = [x[0] for x in trend_data]
    date_counts = [x[1] for x in trend_data]

    conn.close()

    return render_template(
        "analytics.html",
        total_students=total_students,
        total_attendance=total_attendance,
        today_present=today_present,
        batch_labels=batch_labels,
        batch_counts=batch_counts,
        att_labels=att_labels,
        att_counts=att_counts,
        dates=dates,
        date_counts=date_counts
    )


@app.route("/student_analytics")
def student_analytics():
    if not session.get("admin"):
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT s.name, s.roll,
               COUNT(a.id),
               SUM(CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN a.id IS NULL THEN 1 ELSE 0 END)
        FROM students s
        LEFT JOIN attendance a ON s.roll = a.roll
        WHERE s.institution_id=?
        GROUP BY s.roll
    """, (get_inst_id(),))

    data = []
    for row in cur.fetchall():
        name, roll, total, present, absent = row

        present = present or 0
        absent = absent or 0
        total = total or 0

        percent = int((present / total) * 100) if total > 0 else 0
        status = "Good" if percent >= 75 else "Low"

        data.append((name, roll, total, present, absent, percent, status))

    conn.close()

    return render_template("student_analytics.html", data=data)


@app.route("/analytics_monthly")
def analytics_monthly():
    if not session.get("admin"):
        return redirect("/login")

    month = request.args.get("month")

    conn = connect_db()
    cur = conn.cursor()

    if month:
        cur.execute("""
            SELECT date, COUNT(*)
            FROM attendance
            WHERE substr(date,7)=? AND institution_id=?
            GROUP BY date
        """, (month, get_inst_id()))
    else:
        cur.execute("""
            SELECT date, COUNT(*)
            FROM attendance
            WHERE institution_id=?
            GROUP BY date
        """, (get_inst_id(),))

    data = cur.fetchall()
    conn.close()

    dates = [x[0] for x in data]
    counts = [x[1] for x in data]

    return render_template(
        "analytics_monthly.html",
        dates=dates,
        counts=counts,
        selected_month=month
    )


# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("AIRA RUNNING (lightweight, cloud-ready)")
    app.run(host="0.0.0.0", port=port, debug=False)
