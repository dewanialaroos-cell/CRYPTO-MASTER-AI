import csv
import io
import json
import os
import secrets
import hashlib
import smtplib
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CRYPTO_APP_DB", ROOT / "app_users.sqlite3"))
DATABASE_URL = os.environ.get("CRYPTO_APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
SCANNER_PATH = ROOT / "Scanner.py"
RESULTS_PATH = ROOT / "crypto_scan_results.csv"
LEARNING_PATH = ROOT / "ai_learning.json"
app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "1") != "0",
    MAX_CONTENT_LENGTH=16 * 1024,
)
STATE_LOCK = threading.Lock()
LEARNING_MERGE_LOCK = threading.Lock()
SCAN_CONCURRENCY = max(1, min(int(os.environ.get("SCAN_CONCURRENCY", "4")), 32))
SCAN_EXECUTOR = ThreadPoolExecutor(max_workers=SCAN_CONCURRENCY, thread_name_prefix="crypto-scan")
scan_state = {"status": "idle", "started_at": None, "finished_at": None, "error": None, "results": [], "run_id": None}
MAX_HISTORY = 20


class PostgresCompatConnection:
    """Small adapter so the app can use PostgreSQL or local SQLite."""
    def __init__(self, url):
        import psycopg
        from psycopg.rows import dict_row
        self._conn = psycopg.connect(url, row_factory=dict_row, connect_timeout=15, prepare_threshold=None)

    def execute(self, query, params=()):
        return self._conn.execute(query.replace("?", "%s"), params)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)


def db():
    if DATABASE_URL:
        return PostgresCompatConnection(DATABASE_URL)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    if DATABASE_URL:
        with db() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY, email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL, email_verified INTEGER NOT NULL DEFAULT 0,
                firebase_uid TEXT UNIQUE, created_at TEXT NOT NULL
            )""")
            con.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS firebase_uid TEXT UNIQUE")
            con.execute("""CREATE TABLE IF NOT EXISTS auth_tokens (
                id BIGSERIAL PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id),
                token_hash TEXT NOT NULL UNIQUE, purpose TEXT NOT NULL, expires_at TEXT NOT NULL,
                used_at TEXT, created_at TEXT NOT NULL
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS scan_runs (
                id BIGSERIAL PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id),
                status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                top_coins_json TEXT NOT NULL DEFAULT '[]', error TEXT
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS shared_ai_learning (
                id INTEGER PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            con.execute("""CREATE TABLE IF NOT EXISTS scan_observations (
                id BIGSERIAL PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id),
                run_id BIGINT NOT NULL REFERENCES scan_runs(id), symbol TEXT NOT NULL,
                scanned_at TEXT NOT NULL, result_json TEXT NOT NULL
            )""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_scan_observations_scanned_at ON scan_observations(scanned_at)")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, email_verified INTEGER NOT NULL DEFAULT 0,
            firebase_uid TEXT UNIQUE, created_at TEXT NOT NULL
        )""")
        user_columns = {row["name"] for row in con.execute("PRAGMA table_info(users)").fetchall()}
        if "email_verified" not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
        if "firebase_uid" not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN firebase_uid TEXT")
        con.execute("""CREATE TABLE IF NOT EXISTS auth_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE, purpose TEXT NOT NULL, expires_at TEXT NOT NULL,
            used_at TEXT, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS scan_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, status TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, top_coins_json TEXT NOT NULL DEFAULT '[]',
            error TEXT, FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS shared_ai_learning (
            id INTEGER PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS scan_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL, symbol TEXT NOT NULL, scanned_at TEXT NOT NULL,
            result_json TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(run_id) REFERENCES scan_runs(id)
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_scan_observations_scanned_at ON scan_observations(scanned_at)")


def utc_now():
    return datetime.now(timezone.utc).isoformat()



def token_hash(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def email_service_ready():
    return bool(os.environ.get("RESEND_API_KEY") and os.environ.get("RESEND_FROM")) or all(
        os.environ.get(k) for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")
    )


def send_auth_email(email, subject, body):
    resend_key = os.environ.get("RESEND_API_KEY")
    resend_sender = os.environ.get("RESEND_FROM")
    if resend_key and resend_sender:
        import requests
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
            json={"from": resend_sender, "to": [email], "subject": subject, "text": body},
            timeout=20,
        )
        if not response.ok:
            app.logger.error("Email provider rejected message (HTTP %s)", response.status_code)
            raise RuntimeError("Email delivery failed.")
        return

    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM", user or "")
    port = int(os.environ.get("SMTP_PORT", "587"))
    if not all([host, user, password, sender]):
        raise RuntimeError("Email service is temporarily unavailable.")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, email
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


def issue_token(user_id, purpose):
    raw = secrets.token_urlsafe(32)
    hours = 24 if purpose == "verify" else 1
    expiry = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    with db() as con:
        con.execute("UPDATE auth_tokens SET used_at=? WHERE user_id=? AND purpose=? AND used_at IS NULL", (utc_now(), user_id, purpose))
        con.execute("INSERT INTO auth_tokens(user_id,token_hash,purpose,expires_at,created_at) VALUES(?,?,?,?,?)",
                    (user_id, token_hash(raw), purpose, expiry, utc_now()))
    return raw


def _firebase_admin_app():
    """Initialize Firebase Admin from a Render environment variable, never from client config."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        if not raw:
            raise RuntimeError("Firebase server verification is not configured. Add FIREBASE_SERVICE_ACCOUNT_JSON in Render.")
        return firebase_admin.initialize_app(credentials.Certificate(json.loads(raw)))


def current_user():
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        try:
            claims = firebase_auth.verify_id_token(header[7:].strip(), app=_firebase_admin_app(), check_revoked=True)
            uid = claims.get("uid")
            email = (claims.get("email") or "").strip().lower()
            if not uid or not email:
                return None
            with db() as con:
                user = con.execute("SELECT id,email FROM users WHERE firebase_uid=?", (uid,)).fetchone()
                if not user:
                    user = con.execute("SELECT id,email FROM users WHERE email=?", (email,)).fetchone()
                    if user:
                        con.execute("UPDATE users SET firebase_uid=?,email_verified=1 WHERE id=?", (uid, user["id"]))
                    else:
                        placeholder = generate_password_hash(secrets.token_urlsafe(48))
                        cur = con.execute(
                            "INSERT INTO users(email,password_hash,email_verified,firebase_uid,created_at) VALUES(?,?,?,?,?) RETURNING id",
                            (email, placeholder, 1, uid, utc_now())
                        )
                        user = con.execute("SELECT id,email FROM users WHERE id=?", (cur.fetchone()["id"],)).fetchone()
                return user
        except Exception:
            app.logger.exception("Firebase ID token verification failed")
            return None
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
                "v6_status": r.get("v6_status", "UNAVAILABLE"),
                "v6_probability": safe_float(r.get("v6_probability")),
                "v6_signal": r.get("v6_signal", "HOLD"),
                "v6_model_version": r.get("v6_model_version", "V6"),
                "v6_paper_only": str(r.get("v6_paper_only", "true")).strip().lower() in ("true", "1", "yes"),
                "v6_timeframe": r.get("v6_timeframe", "1h"),
                "v6_domain_probabilities": r.get("v6_domain_probabilities", ""),
                "v6_specialist_count": int(safe_float(r.get("v6_specialist_count")) or 0),
                "ensemble_signal": r.get("ensemble_signal", r.get("signal", "NO TRADE")),
                "ensemble_v6_adjustment": safe_float(r.get("ensemble_v6_adjustment")),
                "ensemble_v6_used": str(r.get("ensemble_v6_used", "false")).strip().lower() in ("true", "1", "yes"),
                "ensemble_specialist_count": int(safe_float(r.get("ensemble_specialist_count")) or 0),
                "learning_total": int(safe_float(r.get("learning_total")) or 0),
                "learning_hit_rate": safe_float(r.get("learning_hit_rate")),
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


def get_shared_learning_payload():
    """Read the canonical learning memory from durable storage."""
    with db() as con:
        row = con.execute("SELECT payload FROM shared_ai_learning WHERE id=1").fetchone()
        if row:
            payload = row["payload"]
        else:
            if LEARNING_PATH.exists():
                try:
                    payload = LEARNING_PATH.read_text(encoding="utf-8")
                    json.loads(payload)
                except Exception:
                    payload = json.dumps({"version": 1, "pending": [], "total": 0, "hits": 0})
            else:
                payload = json.dumps({"version": 1, "pending": [], "total": 0, "hits": 0})
            con.execute(
                "INSERT INTO shared_ai_learning(id,payload,updated_at) VALUES(1,?,?) ON CONFLICT(id) DO NOTHING",
                (payload, utc_now()),
            )
            row = con.execute("SELECT payload FROM shared_ai_learning WHERE id=1").fetchone()
            payload = row["payload"]
    json.loads(payload)
    return payload


def _prediction_key(item):
    return (str(item.get("symbol", "")), str(item.get("signal", "")), str(item.get("created", "")))


def persist_shared_learning_and_observations(run_id, user_id, learning_path, results_path, base_payload):
    """Save each scan and merge its learning delta without overwriting concurrent scans."""
    local = json.loads(learning_path.read_text(encoding="utf-8"))
    base = json.loads(base_payload)
    with LEARNING_MERGE_LOCK:
        with db() as con:
            query = "SELECT payload FROM shared_ai_learning WHERE id=1"
            if DATABASE_URL:
                query += " FOR UPDATE"
            row = con.execute(query).fetchone()
            current = json.loads(row["payload"]) if row else {"version": 1, "pending": [], "total": 0, "hits": 0}
            base_keys = {_prediction_key(x) for x in base.get("pending", [])}
            local_keys = {_prediction_key(x) for x in local.get("pending", [])}
            resolved_keys = base_keys - local_keys
            pending = [x for x in current.get("pending", []) if _prediction_key(x) not in resolved_keys]
            pending_keys = {_prediction_key(x) for x in pending}
            for item in local.get("pending", []):
                key = _prediction_key(item)
                if key not in base_keys and key not in pending_keys:
                    pending.append(item)
                    pending_keys.add(key)
            current["version"] = max(int(current.get("version", 1)), int(local.get("version", 1)))
            current["total"] = int(current.get("total", 0)) + max(0, int(local.get("total", 0)) - int(base.get("total", 0)))
            current["hits"] = int(current.get("hits", 0)) + max(0, int(local.get("hits", 0)) - int(base.get("hits", 0)))
            current["pending"] = pending[-5000:]
            payload = json.dumps(current, ensure_ascii=False)
            con.execute(
                "INSERT INTO shared_ai_learning(id,payload,updated_at) VALUES(1,?,?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (payload, utc_now()),
            )
    if not results_path.exists():
        return 0
    with results_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scanned_at = utc_now()
    inserted = 0
    with db() as con:
        for row in rows:
            symbol = str(row.get("symbol") or "").strip()
            if not symbol:
                continue
            con.execute(
                "INSERT INTO scan_observations(user_id,run_id,symbol,scanned_at,result_json) VALUES(?,?,?,?,?)",
                (user_id, run_id, symbol, scanned_at, json.dumps(row, ensure_ascii=False)),
            )
            inserted += 1
    return inserted


def scan_worker(run_id, user_id):
    with db() as con:
        con.execute("UPDATE scan_runs SET status=? WHERE id=?", ("running", run_id))
    try:
        base_payload = get_shared_learning_payload()
        # Per-run cwd prevents CSV and ai_learning.json collisions between users.
        with tempfile.TemporaryDirectory(prefix=f"crypto-scan-{run_id}-") as job_dir_name:
            job_dir = Path(job_dir_name)
            job_learning = job_dir / "ai_learning.json"
            job_results = job_dir / "crypto_scan_results.csv"
            job_learning.write_text(base_payload, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCANNER_PATH)],
                cwd=str(job_dir), capture_output=True, text=True, timeout=900, check=False,
                env=os.environ.copy()
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "Scanner exited with an error")[-2500:]
                raise RuntimeError(detail)
            persist_shared_learning_and_observations(run_id, user_id, job_learning, job_results, base_payload)
            results = parse_top_coins(job_results)
            finished = utc_now()
            with db() as con:
                con.execute("UPDATE scan_runs SET status=?,finished_at=?,top_coins_json=? WHERE id=? AND user_id=?",
                            ("completed", finished, json.dumps(results), run_id, user_id))
    except Exception as e:
        finished = utc_now()
        message = str(e)[-2500:]
        with db() as con:
            con.execute("UPDATE scan_runs SET status=?,finished_at=?,error=? WHERE id=? AND user_id=?",
                        ("failed", finished, message, run_id, user_id))
        app.logger.exception("Scan run %s failed", run_id)


QUEUE_START_LOCK = threading.Lock()
QUEUE_WORKER_STARTED = False


def scan_queue_worker():
    """Claim durable scan jobs and execute several isolated scans concurrently."""
    in_flight = set()
    while True:
        try:
            in_flight = {future for future in in_flight if not future.done()}
            slots = SCAN_CONCURRENCY - len(in_flight)
            for _ in range(max(0, slots)):
                with db() as con:
                    row = con.execute(
                        "SELECT id,user_id FROM scan_runs WHERE status='queued' ORDER BY id ASC LIMIT 1"
                    ).fetchone()
                    if not row:
                        break
                    claimed = con.execute(
                        "UPDATE scan_runs SET status='claimed' WHERE id=? AND status='queued'",
                        (row["id"],),
                    )
                    if claimed.rowcount != 1:
                        continue
                    job = (row["id"], row["user_id"])
                in_flight.add(SCAN_EXECUTOR.submit(scan_worker, job[0], job[1]))
        except Exception:
            app.logger.exception("Scan queue worker error")
        time.sleep(0.5)


def ensure_scan_queue_worker():
    global QUEUE_WORKER_STARTED
    with QUEUE_START_LOCK:
        if not QUEUE_WORKER_STARTED:
            thread = threading.Thread(target=scan_queue_worker, name="crypto-scan-queue", daemon=True)
            thread.start()
            QUEUE_WORKER_STARTED = True


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
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 10 or len(password) > 128:
        return jsonify({"error": "Password must be between 10 and 128 characters."}), 400
    # Email verification is used when SMTP is configured. If SMTP is not configured,
    # allow password-based signup/login instead of leaving users locked out.
    smtp_ready = email_service_ready()
    try:
        with db() as con:
            cur = con.execute("INSERT INTO users(email,password_hash,email_verified,created_at) VALUES(?,?,?,?) RETURNING id",
                              (email, generate_password_hash(password), 0 if smtp_ready else 1, utc_now()))
            user_id = cur.fetchone()["id"]
        if smtp_ready:
            raw = issue_token(user_id, "verify")
            base = request.host_url.rstrip("/")
            try:
                send_auth_email(email, "Verify your Crypto Master AI email",
                                f"Open this link within 24 hours to verify your account:\n\n{base}/verify-email?token={raw}")
            except Exception:
                app.logger.exception("Verification email delivery failed")
                return jsonify({"error": "Your account was created, but the verification email could not be sent. Please try again later or contact support."}), 503
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists. Please log in."}), 409
    except Exception as exc:
        # PostgreSQL unique-constraint violation (SQLSTATE 23505) or SQLite duplicate email.
        if getattr(exc, "sqlstate", None) == "23505":
            return jsonify({"error": "An account with this email already exists. Please log in."}), 409
        app.logger.exception("Signup failed")
        return jsonify({"error": "We could not create your account. Please try again later."}), 503
    message = ("Sign-up successful. Please verify your email and log in." if smtp_ready
               else "Sign-up successful. You can now log in with your password.")
    return jsonify({"ok": True, "message": message}), 201



@app.get("/verify-email")
def verify_email():
    raw = request.args.get("token", "")
    if not raw:
        return "Verification token is missing.", 400
    hashed = token_hash(raw)
    with db() as con:
        row = con.execute("SELECT user_id,expires_at,used_at FROM auth_tokens WHERE token_hash=? AND purpose='verify'", (hashed,)).fetchone()
        if not row or row["used_at"] or row["expires_at"] < utc_now():
            return "This verification link is invalid or has expired.", 400
        con.execute("UPDATE users SET email_verified=1 WHERE id=?", (row["user_id"],))
        con.execute("UPDATE auth_tokens SET used_at=? WHERE token_hash=?", (utc_now(), hashed))
    return "Email verified. Return to Crypto Master AI and log in.", 200


@app.post("/api/forgot-password")
def forgot_password():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    if len(email) > 254 or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        return jsonify({"error": "Please enter a valid email address."}), 400

    # Never report success when the app cannot actually send a reset email.
    smtp_ready = email_service_ready()
    if not smtp_ready:
        return jsonify({"error": "Password reset is temporarily unavailable. Please try again later."}), 503

    try:
        with db() as con:
            user = con.execute("SELECT id,email FROM users WHERE email=?", (email,)).fetchone()
        if user:
            raw = issue_token(user["id"], "reset")
            base = request.host_url.rstrip("/")
            send_auth_email(email, "Reset your Crypto Master AI password",
                            f"Open this link within 1 hour to reset your password:\\n\\n{base}/reset-password?token={raw}")
        # Keep the response generic so it does not reveal whether an account exists.
        return jsonify({"ok": True, "message": "If an account exists for this email, a password reset link has been sent."})
    except Exception:
        app.logger.exception("Password reset request failed")
        return jsonify({"error": "We could not send the password reset email. Please try again later."}), 503


@app.get("/reset-password")
def reset_password_page():
    token = request.args.get("token", "")
    if not token:
        return "Reset token is missing.", 400
    safe_token = token.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reset password — Crypto Master AI</title><style>body{background:#080a0d;color:#f5f5f5;font:16px sans-serif;display:grid;place-items:center;min-height:90vh}.card{max-width:420px;padding:28px;border:1px solid #a98535;border-radius:18px}input,button{box-sizing:border-box;width:100%;padding:12px;margin:10px 0;border-radius:8px}button{background:#d5aa4c;border:0;font-weight:bold}</style></head><body><div class="card"><h1>Reset password</h1><p>Your new password must be at least 10 characters.</p><form id="f"><input id="p" type="password" minlength="10" maxlength="128" required placeholder="New password"><button>Save new password</button></form><p id="m"></p><a href="/">Return to login</a></div><script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();let r=await fetch('/api/reset-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:""" + JSON.stringify(safe_token) + """,password:document.getElementById('p').value})});let d=await r.json();document.getElementById('m').textContent=d.message||d.error||'Request complete';};</script></body></html>""", 200


@app.post("/api/reset-password")
def reset_password():
    payload = request.get_json(silent=True) or {}
    raw, password = str(payload.get("token", "")), str(payload.get("password", ""))
    if len(password) < 10 or len(password) > 128:
        return jsonify({"error": "Password must be between 10 and 128 characters."}), 400
    hashed = token_hash(raw)
    with db() as con:
        row = con.execute("SELECT user_id,expires_at,used_at FROM auth_tokens WHERE token_hash=? AND purpose='reset'", (hashed,)).fetchone()
        if not row or row["used_at"] or row["expires_at"] < utc_now():
            return jsonify({"error": "This reset link is invalid or has expired."}), 400
        con.execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(password), row["user_id"]))
        con.execute("UPDATE auth_tokens SET used_at=? WHERE user_id=? AND purpose='reset' AND used_at IS NULL", (utc_now(), row["user_id"]))
    return jsonify({"ok": True, "message": "Password reset successfully. You can now log in."})


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    with db() as con:
        user = con.execute("SELECT id,email,password_hash,email_verified FROM users WHERE email=?", (email,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Incorrect email or password."}), 401
    smtp_ready = all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"))
    if not user["email_verified"] and smtp_ready:
        return jsonify({"error": "Please verify your email before logging in. Check your inbox and spam folder."}), 403
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
        return jsonify({"error": "Your security token expired. Please refresh the page."}), 403
    if not SCANNER_PATH.exists():
        return jsonify({"error": "Scanner.py was not found in the repository."}), 500
    started = utc_now()
    try:
        with db() as con:
            cur = con.execute("INSERT INTO scan_runs(user_id,status,started_at,top_coins_json) VALUES(?,?,?,?) RETURNING id",
                              (user["id"], "queued", started, "[]"))
            run_id = cur.fetchone()["id"]
        # Every customer's request is saved and queued; one scanner processes jobs
        # sequentially because the scanner's CSV and JSON memory are shared files.
        ensure_scan_queue_worker()
        return jsonify({"ok": True, "run_id": run_id, "status": "queued"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/coin/<path:symbol>")
@login_required
def coin_details(symbol):
    """Return a coin's latest saved details belonging only to the signed-in user."""
    user = current_user()
    try:
        with db() as con:
            saved = con.execute(
                """SELECT so.result_json FROM scan_observations so
                   JOIN scan_runs sr ON sr.id=so.run_id
                   WHERE so.user_id=? AND sr.status='completed' AND UPPER(so.symbol)=?
                   ORDER BY so.id DESC LIMIT 1""",
                (user["id"], symbol.strip().upper()),
            ).fetchone()
        if not saved:
            return jsonify({"error": "No recent scan data was found for this coin in your account. Please run the scanner again."}), 404
        row = json.loads(saved["result_json"])
        details = {}
        for key, value in row.items():
            if value is None or value == "":
                details[key] = None
                continue
            try:
                number = float(value)
                details[key] = number if number == number and abs(number) != float("inf") else value
            except (ValueError, TypeError):
                details[key] = value
        return jsonify({"symbol": row.get("symbol", symbol), "details": details})
    except Exception:
        app.logger.exception("Could not load saved coin details")
        return jsonify({"error": "Could not load coin details."}), 500


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "Crypto Master AI Web App"})


init_db()
ensure_scan_queue_worker()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
