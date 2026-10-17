import csv
import io
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CRYPTO_APP_DB", ROOT / "app_users.sqlite3"))
SCANNER_PATH = ROOT / "Scanner.py"
RESULTS_PATH = ROOT / "crypto_scan_results.csv"
app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "1") != "0",
    MAX_CONTENT_LENGTH=16 * 1024,
)
SCAN_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
scan_state = {"status": "idle", "started_at": None, "finished_at": None, "error": None, "results": [], "run_id": None}
MAX_HISTORY = 20


def db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS scan_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            top_coins_json TEXT NOT NULL DEFAULT '[]',
            error TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with db() as con:
        return con.execute("SELECT id,email FROM users WHERE id=?", (uid,)).fetchone()


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not current_user():
            return jsonify({"error": "Please log in first."}), 401
        return fn(*args, **kwargs)
    return wrapped


def parse_top_coins(path, limit=10):
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        actionable = [r for r in rows if r.get("signal") in ("LONG", "SHORT")]
        actionable.sort(key=lambda r: float(r.get("signal_strength") or 0), reverse=True)
        output = []
        for r in actionable[:limit]:
            output.append({
                "symbol": r.get("symbol", "—"),
                "signal": r.get("signal", "NO TRADE"),
                "price": safe_float(r.get("price")),
                "strength": safe_float(r.get("signal_strength")),
                "confidence": safe_float(r.get("confidence")),
                "stop_loss": safe_float(r.get("stop_loss")),
                "take_profit": safe_float(r.get("take_profit")),
                "exchange": r.get("analysis_exchange", "—"),
                "timestamp": r.get("timestamp", ""),
                "reason": r.get("reason", ""),
            })
        return output
    except Exception:
        return []


def safe_float(value):
    try:
        v = float(value)
        return v if v == v and abs(v) != float("inf") else None
    except (TypeError, ValueError):
        return None


def scan_worker(run_id, user_id):
    started = utc_now()
    with STATE_LOCK:
        scan_state.update(status="running", started_at=started, finished_at=None, error=None, results=[], run_id=run_id)
    try:
        # A single scanner process is allowed at a time because the scanner shares
        # its learning file and CSV output. No exchange orders are placed by this app.
        completed = subprocess.run(
            [sys.executable, str(SCANNER_PATH)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=900, check=False,
            env=os.environ.copy()
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Scanner exited with an error")[-2500:]
            raise RuntimeError(detail)
        results = parse_top_coins(RESULTS_PATH)
        finished = utc_now()
        with db() as con:
            con.execute("UPDATE scan_runs SET status=?,finished_at=?,top_coins_json=? WHERE id=? AND user_id=?",
                        ("completed", finished, json.dumps(results), run_id, user_id))
        with STATE_LOCK:
            scan_state.update(status="completed", finished_at=finished, results=results, error=None)
    except Exception as e:
        finished = utc_now()
        message = str(e)[-2500:]
        with db() as con:
            con.execute("UPDATE scan_runs SET status=?,finished_at=?,error=? WHERE id=? AND user_id=?",
                        ("failed", finished, message, run_id, user_id))
        with STATE_LOCK:
            scan_state.update(status="failed", finished_at=finished, error=message, results=[])
    finally:
        SCAN_LOCK.release()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/me")
def me():
    user = current_user()
    if not user:
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "email": user["email"], "csrf": session.setdefault("csrf", secrets.token_urlsafe(24))})


@app.post("/api/signup")
def signup():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    if len(email) > 254 or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        return jsonify({"error": "Valid email address enter karein."}), 400
    if len(password) < 10 or len(password) > 128:
        return jsonify({"error": "Password kam az kam 10 characters ka ho."}), 400
    try:
        with db() as con:
            cur = con.execute("INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
                              (email, generate_password_hash(password), utc_now()))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "Is email se account pehle se mojood hai. Login karein."}), 409
    session.clear()
    session["user_id"] = user_id
    session["csrf"] = secrets.token_urlsafe(24)
    return jsonify({"ok": True, "email": email, "csrf": session["csrf"]})


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    with db() as con:
        user = con.execute("SELECT id,email,password_hash FROM users WHERE email=?", (email,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Email ya password ghalat hai."}), 401
    session.clear()
    session["user_id"] = user["id"]
    session["csrf"] = secrets.token_urlsafe(24)
    return jsonify({"ok": True, "email": user["email"], "csrf": session["csrf"]})


@app.post("/api/logout")
@login_required
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/status")
@login_required
def status():
    user = current_user()
    with db() as con:
        last = con.execute("SELECT id,status,started_at,finished_at,top_coins_json,error FROM scan_runs WHERE user_id=? ORDER BY id DESC LIMIT 1",
                            (user["id"],)).fetchone()
    if not last:
        return jsonify({"status": "idle", "results": []})
    return jsonify({"run_id": last["id"], "status": last["status"], "started_at": last["started_at"],
                    "finished_at": last["finished_at"], "results": json.loads(last["top_coins_json"] or "[]"),
                    "error": last["error"]})


@app.post("/api/run")
@login_required
def run_scan():
    user = current_user()
    payload = request.get_json(silent=True) or {}
    if not secrets.compare_digest(str(payload.get("csrf", "")), str(session.get("csrf", ""))):
        return jsonify({"error": "Security token expired. Page refresh karein."}), 403
    if not SCAN_LOCK.acquire(blocking=False):
        return jsonify({"error": "AI scan pehle se chal raha hai. Complete hone ka intezar karein."}), 409
    if not SCANNER_PATH.exists():
        SCAN_LOCK.release()
        return jsonify({"error": "Scanner.py repository mein nahi mila."}), 500
    started = utc_now()
    try:
        with db() as con:
            cur = con.execute("INSERT INTO scan_runs(user_id,status,started_at,top_coins_json) VALUES(?,?,?,?)",
                              (user["id"], "running", started, "[]"))
            run_id = cur.lastrowid
        threading.Thread(target=scan_worker, args=(run_id, user["id"]), daemon=True).start()
        return jsonify({"ok": True, "run_id": run_id, "status": "running"})
    except Exception as e:
        SCAN_LOCK.release()
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "Crypto Master AI Web App"})


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
