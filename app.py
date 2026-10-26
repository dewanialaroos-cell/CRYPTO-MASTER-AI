import csv
import io
import json
import os
import secrets
import hashlib
import smtplib
import sqlite3
import subprocess
import requests
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
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
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


def parse_top_coins(path, limit=None):
    """Return every coin row from this user's latest scan, strongest actionable signals first."""
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rows.sort(
            key=lambda r: (
                0 if str(r.get("signal", "")).upper() in ("LONG", "SHORT") else 1,
                -(safe_float(r.get("signal_strength")) or 0),
            )
        )
        output = []
        for r in rows:
            symbol = str(r.get("symbol") or "").strip()
            if not symbol:
                continue
            output.append({
                "symbol": symbol,
                "signal": str(r.get("signal") or "NO TRADE").upper(),
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
        return output[:limit] if limit else output
    except Exception:
        app.logger.exception("Could not parse scan results from %s", path)
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
    """Merge shared learning, but persist coin data only for the user's current run."""
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
            base_resolved = {str(x.get("id")) for x in base.get("resolved", []) if x.get("id")}
            current_resolved = list(current.get("resolved", []))
            current_resolved_ids = {str(x.get("id")) for x in current_resolved if x.get("id")}
            for outcome in local.get("resolved", []):
                outcome_id = str(outcome.get("id") or "")
                if not outcome_id or outcome_id in base_resolved or outcome_id in current_resolved_ids:
                    continue
                current_resolved.append(outcome)
                current_resolved_ids.add(outcome_id)
                current["total"] = int(current.get("total", 0)) + 1
                if bool(outcome.get("hit")):
                    current["hits"] = int(current.get("hits", 0)) + 1
            current["resolved"] = current_resolved[-10000:]
            current["version"] = max(int(current.get("version", 1)), int(local.get("version", 1)))
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
        # A newer scan may have replaced this run while the scanner was working.
        # Ignore stale workers so they cannot re-create deleted coin results.
        current = con.execute("SELECT id FROM scan_runs WHERE id=? AND user_id=?", (run_id, user_id)).fetchone()
        if not current:
            return 0
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
        claimed = con.execute(
            "UPDATE scan_runs SET status=? WHERE id=? AND user_id=? AND status IN ('queued','claimed')",
            ("running", run_id, user_id),
        )
        if claimed.rowcount != 1:
            return  # This run was replaced by a newer scan or already claimed.
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


def github_actions_enabled():
    """Use GitHub Actions only when both required Render secrets are configured."""
    return bool(
        os.environ.get("GITHUB_ACTIONS_TOKEN", "").strip()
        and os.environ.get("CRYPTO_SCAN_CALLBACK_SECRET", "").strip()
    )


def dispatch_github_scan(run_id, user_id):
    token = os.environ["GITHUB_ACTIONS_TOKEN"].strip()
    repository = os.environ.get("GITHUB_REPOSITORY", "dewanialaroos-cell/CRYPTO-MASTER-AI").strip()
    ref = os.environ.get("GITHUB_ACTIONS_REF", "main").strip()
    url = f"https://api.github.com/repos/{repository}/actions/workflows/scan.yml/dispatches"
    response = requests.post(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": ref, "inputs": {"run_id": str(run_id), "user_id": str(user_id)}},
        timeout=20,
    )
    if response.status_code != 204:
        detail = (response.text or "GitHub workflow dispatch failed")[:1000]
        raise RuntimeError(f"GitHub Actions dispatch failed (HTTP {response.status_code}): {detail}")


def scan_queue_worker():
    """Claim one durable job at a time; use GitHub Actions when configured."""
    in_flight = set()
    # A Render restart kills local subprocesses. Re-queue interrupted local jobs
    # once at process startup instead of leaving their database status stuck forever.
    if not github_actions_enabled():
        try:
            with db() as con:
                con.execute(
                    "UPDATE scan_runs SET status='queued',started_at=?,finished_at=NULL,error=NULL "
                    "WHERE status IN ('claimed','running')",
                    (utc_now(),),
                )
        except Exception:
            app.logger.exception("Could not recover interrupted local scans")
    while True:
        try:
            if github_actions_enabled():
                # A dispatched/running GitHub workflow owns the single scan slot.
                with db() as con:
                    active = con.execute(
                        "SELECT id,started_at,status FROM scan_runs "
                        "WHERE status IN ('dispatched','running') ORDER BY id ASC LIMIT 1"
                    ).fetchone()
                    if active:
                        try:
                            started = datetime.fromisoformat(str(active["started_at"]).replace("Z", "+00:00"))
                            if started.tzinfo is None:
                                started = started.replace(tzinfo=timezone.utc)
                            stale = datetime.now(timezone.utc) - started > timedelta(minutes=20)
                        except Exception:
                            stale = False
                        if stale:
                            con.execute(
                                "UPDATE scan_runs SET status='failed',finished_at=?,error=? "
                                "WHERE id=? AND status IN ('dispatched','running')",
                                (utc_now(), "GitHub Actions did not report scan completion within 20 minutes. Check the workflow logs.", active["id"]),
                            )
                        else:
                            time.sleep(0.5)
                            continue

                    queued = con.execute(
                        "SELECT id,user_id FROM scan_runs WHERE status='queued' ORDER BY id ASC LIMIT 1"
                    ).fetchone()
                    if not queued:
                        time.sleep(0.5)
                        continue
                    claimed = con.execute(
                        "UPDATE scan_runs SET status='dispatched',started_at=?,finished_at=NULL,error=NULL "
                        "WHERE id=? AND status='queued'",
                        (utc_now(), queued["id"]),
                    )
                    if claimed.rowcount != 1:
                        continue
                    job = (queued["id"], queued["user_id"])
                try:
                    dispatch_github_scan(job[0], job[1])
                    app.logger.info("Dispatched GitHub Actions scan run_id=%s user_id=%s", job[0], job[1])
                except Exception as exc:
                    with db() as con:
                        con.execute(
                            "UPDATE scan_runs SET status='failed',finished_at=?,error=? WHERE id=? AND status='dispatched'",
                            (utc_now(), str(exc)[-2500:], job[0]),
                        )
                    app.logger.exception("Could not dispatch GitHub Actions scan run_id=%s", job[0])
                continue

            # Safe local fallback remains available until Render secrets are configured.
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



@app.get("/faq")
def faq_page():
    return render_template("info.html", page_title="Crypto Master AI FAQ", page_label="FREQUENTLY ASKED QUESTIONS", page_intro="Straight answers about using the scanner, reading its outputs, and understanding its limits.", sections=[
        {"title":"Is Crypto Master AI free to use?","body":"Availability and any account limits depend on the current service configuration. The website may change its features or limits over time; check the dashboard for the current experience."},
        {"title":"Does the scanner guarantee profitable trades?","body":"No. Crypto markets are uncertain and a signal can be wrong. Confidence scores and model rankings are internal estimates, not guarantees or personalized recommendations."},
        {"title":"Does the dashboard place trades automatically?","body":"The dashboard is for market analysis and showing scan results. Do not assume that it places orders or manages an exchange account. Verify the current product features before relying on any workflow."},
        {"title":"Why might a scan take time or show fewer coins?","body":"Exchange API availability, rate limits, network issues, missing candles, and market-data coverage can affect scan time and results. Try again later if a provider is temporarily unavailable."},
        {"title":"What do SL and TP mean?","body":"SL means Stop-Loss and TP means Take-Profit. When displayed, these are reference levels generated by the scanner, not guaranteed execution prices. Fees, slippage, liquidity, and volatility can change outcomes."},
        {"title":"Can I rely on historical backtests?","body":"Backtests are useful for research but can be misleading if data is incomplete or costs and execution are not modelled realistically. Past results do not predict future performance; test independently and use paper trading cautiously."},
        {"title":"How should I protect my account?","body":"Use a unique strong password, keep credentials private, sign out on shared devices, and never share wallet seed phrases or private keys. Contact the site operator through the published contact channel for account support."}
    ])

@app.get("/contact")
def contact_page():
    return render_template("info.html", page_title="Contact Crypto Master AI", page_label="CONTACT", page_intro="For account, privacy, or service questions, use the official contact details published by the website operator.", sections=[
        {"title":"Account and technical support","body":"If you encounter an account or scanner issue, include the page you were using, the approximate time, and a brief description. Never include your password, authentication codes, API secrets, wallet keys, or seed phrases."},
        {"title":"Privacy requests","body":"For privacy questions or requests about personal data, contact the website operator using the public support email listed in the site's official business/contact details."},
        {"title":"Operator contact details","body":"The operator's public support email has not yet been configured in this repository. Before public launch and AdSense review, add a real, monitored contact email and update this page and the Privacy Policy. Do not publish an email address that you do not control."}
    ])

@app.get("/robots.txt")
def robots_txt():
    return ("User-agent: *\\nAllow: /\\nDisallow: /api/\\nDisallow: /dashboard\\nSitemap: https://crypto-master-ai.onrender.com/sitemap.xml\\n", 200, {"Content-Type": "text/plain; charset=utf-8"})

@app.get("/sitemap.xml")
def sitemap_xml():
    urls = [
        ("https://crypto-master-ai.onrender.com/", "weekly"),
        ("https://crypto-master-ai.onrender.com/about", "monthly"),
        ("https://crypto-master-ai.onrender.com/scanner-guide", "monthly"),
        ("https://crypto-master-ai.onrender.com/risk-management", "monthly"),
        ("https://crypto-master-ai.onrender.com/model-transparency", "monthly"),
        ("https://crypto-master-ai.onrender.com/privacy", "monthly"),
        ("https://crypto-master-ai.onrender.com/terms", "monthly"),
        ("https://crypto-master-ai.onrender.com/faq", "monthly"),
        ("https://crypto-master-ai.onrender.com/contact", "monthly"),
    ]
    items = "".join(f"<url><loc>{loc}</loc><changefreq>{freq}</changefreq></url>" for loc, freq in urls)
    xml = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + items + "</urlset>"
    return (xml, 200, {"Content-Type": "application/xml; charset=utf-8"})

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/about")
def about_page():
    return render_template("info.html", page_title="About Crypto Master AI", page_label="ABOUT US", page_intro="An independent market-analysis workspace built to help users inspect crypto-market signals.", sections=[
        {"title":"What Crypto Master AI does","body":"Crypto Master AI scans supported cryptocurrency markets and presents ranked candidates with technical indicators, market context, LONG/SHORT/NO TRADE labels, and risk levels when available. It is an analysis tool, not a broker or exchange."},
        {"title":"How to use it","body":"Create an account, open the dashboard, and select Run Crypto Master AI. When a scan finishes, review the coin cards and open a coin to inspect available details. Check the scan timestamp and confirm current prices independently before making decisions."},
        {"title":"What the results mean","body":"A LONG label indicates that the scanner's rules currently favour a bullish setup; SHORT indicates a bearish setup; NO TRADE means the system does not show a sufficiently clear setup. Confidence and signal-strength values are model estimates, not the probability that a trade will make money."},
        {"title":"Independence and limitations","body":"Crypto Master AI is not affiliated with, endorsed by, or an official product of any cryptocurrency exchange. Data availability, exchange restrictions, API limits, market liquidity, slippage, and model errors can affect results. We do not promise profits or guaranteed accuracy."}
    ])

@app.get("/scanner-guide")
def scanner_guide_page():
    return render_template("info.html", page_title="AI Scanner Guide", page_label="USER GUIDE", page_intro="Understand the scanner output before using it to research a market.", sections=[
        {"title":"1. Start a scan","body":"Sign in and press Run Crypto Master AI. The service collects available market data and runs its configured analysis. Scan duration and coin coverage can change with exchange availability, rate limits, and network conditions."},
        {"title":"2. LONG, SHORT, and NO TRADE","body":"LONG means the combined rules currently lean bullish; SHORT means they lean bearish; NO TRADE means the current evidence does not meet the system's trade criteria. These are research labels, not instructions or guarantees."},
        {"title":"3. Signal strength and confidence","body":"These numbers summarize the scanner's internal scoring. They are not calibrated promises of success, and scores from different coins or market conditions may not be directly comparable. A high score can still lose money."},
        {"title":"4. Stop-Loss and Take-Profit","body":"When shown, Stop-Loss (SL) and Take-Profit (TP) are model-generated reference levels. They may be missing or unsuitable for low-liquidity markets, fast-moving prices, or different exchange prices. Fees, spread, slippage, funding, and execution can change actual outcomes."},
        {"title":"5. Check before acting","body":"Confirm the symbol and exchange, inspect live liquidity and price, understand the position size and maximum loss, and verify the setup independently. Do not treat a scan as personalized financial advice. This website does not place trades for you."}
    ])

@app.get("/risk-management")
def risk_management_page():
    return render_template("info.html", page_title="Crypto Risk Management", page_label="RISK EDUCATION", page_intro="Crypto assets are volatile. Risk controls can limit exposure, but they cannot eliminate losses.", sections=[
        {"title":"Never risk money you cannot afford to lose","body":"Crypto trading can cause rapid and substantial losses. Do not use borrowed money, emergency savings, rent, or funds needed for essential expenses."},
        {"title":"Understand position size","body":"Decide the maximum amount you are willing to lose before considering a trade. Position size should reflect the distance to the invalidation or stop level, account size, fees, and the possibility of slippage."},
        {"title":"Stop-Loss is not a guarantee","body":"A stop order may execute at a worse price during gaps, volatility, thin liquidity, or exchange outages. Some instruments and order types have additional risks."},
        {"title":"Use Take-Profit carefully","body":"A target is a planning reference, not a promise that the market will reach it. Consider whether the expected reward justifies the possible loss after fees and funding."},
        {"title":"Avoid excessive leverage","body":"Leverage magnifies gains and losses and can trigger liquidation before a market recovers. Understand maintenance margin, liquidation price, funding costs, and exchange rules before using derivatives."},
        {"title":"Keep a trading journal","body":"Record the reason for each decision, entry and exit assumptions, costs, and outcome. Evaluate a strategy over many trades and different market conditions rather than relying on a few winning examples."}
    ])

@app.get("/model-transparency")
def model_transparency_page():
    return render_template("info.html", page_title="AI Model Transparency", page_label="METHODOLOGY & LIMITATIONS", page_intro="The scanner combines available market data and model scores; its outputs can be wrong or incomplete.", sections=[
        {"title":"Data sources and coverage","body":"The scanner uses supported exchange market endpoints and any configured market-data providers. Coverage varies by asset, timeframe, endpoint availability, API limits, and provider configuration. Missing data can make an analysis incomplete."},
        {"title":"Specialist scores and ensemble","body":"Where enabled, the V6 model and specialist domains contribute model estimates to the wider scanner. A model being connected does not establish that it has a reliable trading edge. Outputs may change as code, features, or data change."},
        {"title":"Backtests are not live performance","body":"Historical tests can be affected by data gaps, survivorship bias, overfitting, fees, slippage, liquidity, and assumptions about execution. Historical performance does not predict future results. Paper-trading results also differ from real execution."},
        {"title":"No guaranteed accuracy or profit","body":"Models may misread regimes, react late, overfit historical patterns, or perform poorly on unseen market conditions. Confidence scores should not be interpreted as verified probabilities unless a separate calibration study demonstrates that."},
        {"title":"Research-only service","body":"Crypto Master AI provides informational market analysis, not personalized financial, investment, tax, or legal advice. It does not guarantee signals, returns, or uninterrupted service and does not execute trades through this dashboard."}
    ])

@app.get("/privacy")
def privacy_page():
    return render_template("info.html", page_title="Privacy Policy", page_label="PRIVACY", page_intro="This notice describes the information the Crypto Master AI website may process. Review it and update it to match the service's actual configuration before public launch.", sections=[
        {"title":"Information you provide","body":"When you create or use an account, the service may process your email address, account identifiers, authentication status, and requests you make to the scanner."},
        {"title":"Usage and scan data","body":"The service may process scan status, timestamps, requested market data, and results associated with your account to operate the dashboard, troubleshoot errors, and improve reliability. Do not submit passwords, API secrets, wallet seed phrases, or other confidential information into support messages."},
        {"title":"Authentication and service providers","body":"The website processes account sign-in through its configured authentication system. Hosting and market-data providers may process technical information necessary to provide their services. Their own privacy terms also apply."},
        {"title":"Cookies, security, and retention","body":"The website may use essential session cookies and technical logs. No website can guarantee absolute security. Data retention depends on the service configuration, storage availability, legal requirements, and operational needs."},
        {"title":"Your choices and contact","body":"Use the account sign-out control to end your current session. For privacy questions or requests about account data, contact the website operator through the contact channel published on this site. Before launch, the operator should add a monitored contact email and a clear process for data requests."},
        {"title":"Advertising cookies and choices","body":"If advertising is enabled, Google and other advertising partners may use cookies or similar technologies to serve, limit, personalize, and measure ads based on visits to this and other websites, subject to applicable settings and law. Visitors can review Google ad personalization choices at https://adssettings.google.com/ and learn about some third-party advertising opt-outs at https://www.aboutads.info/. The operator must update this notice to identify the actual ad partners used and provide any required consent controls before displaying ads."},
        {"title":"Important publication note","body":"This is a starter privacy notice, not legal advice. The site operator must verify that it accurately describes actual data collection, retention, analytics, cookies, advertising partners, and applicable laws. Update this page before enabling advertising or inviting the public."}
    ])

@app.get("/terms")
def terms_page():
    return render_template("info.html", page_title="Terms of Use", page_label="TERMS", page_intro="By using Crypto Master AI, you agree to use the service responsibly and understand its limitations.", sections=[
        {"title":"Research only; no financial advice","body":"The website provides automated market analysis for information and education. It is not financial, investment, legal, or tax advice, and it does not recommend a transaction suitable for your personal circumstances."},
        {"title":"No guarantee","body":"Market data can be delayed, incomplete, inaccurate, or unavailable. Signals, model outputs, stop-loss levels, take-profit levels, and confidence scores can be wrong. No profit, accuracy, uptime, or availability is guaranteed."},
        {"title":"Your responsibility","body":"You are responsible for verifying information, complying with laws and exchange rules, protecting account credentials, and deciding whether any action is appropriate. Never share passwords, private keys, or seed phrases."},
        {"title":"Acceptable use","body":"Do not attempt to disrupt the service, bypass access controls, abuse APIs, submit malicious content, or use the website for unlawful activity. Access may be limited to protect the service and its users."},
        {"title":"Availability and changes","body":"Features may change or be unavailable because of maintenance, provider restrictions, exchange API changes, or infrastructure limits. We may update these terms and the website as the service evolves."},
        {"title":"Contact and review","body":"The operator should publish a monitored contact email before public launch. This starter page is not legal advice and should be reviewed for the operator's jurisdiction and actual business practices."}
    ])


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


def verify_scan_callback_secret():
    expected = os.environ.get("CRYPTO_SCAN_CALLBACK_SECRET", "").strip()
    supplied = request.headers.get("X-Crypto-Scan-Secret", "").strip()
    return bool(expected and supplied and secrets.compare_digest(expected, supplied))


@app.get("/api/github-scan-input")
def github_scan_input():
    """Private workflow endpoint: provide the canonical learning snapshot for one queued job."""
    if not verify_scan_callback_secret():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        run_id = int(request.args.get("run_id", ""))
        user_id = int(request.args.get("user_id", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid scan identifiers"}), 400
    with db() as con:
        run = con.execute(
            "SELECT id,status FROM scan_runs WHERE id=? AND user_id=?",
            (run_id, user_id),
        ).fetchone()
        if not run or run["status"] not in ("dispatched", "running"):
            return jsonify({"error": "Scan job is not active"}), 409
        con.execute(
            "UPDATE scan_runs SET status='running' WHERE id=? AND user_id=? AND status IN ('dispatched','running')",
            (run_id, user_id),
        )
    try:
        return jsonify({"run_id": run_id, "user_id": user_id, "learning_payload": get_shared_learning_payload()})
    except Exception:
        app.logger.exception("Could not prepare learning payload for GitHub scan %s", run_id)
        return jsonify({"error": "Could not prepare scanner input"}), 500


@app.post("/api/github-scan-callback")
def github_scan_callback():
    """Receive an authenticated completion payload from the GitHub Actions scanner."""
    if not verify_scan_callback_secret():
        return jsonify({"error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    try:
        run_id = int(payload.get("run_id"))
        user_id = int(payload.get("user_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid scan identifiers"}), 400
    status_value = str(payload.get("status", "")).lower()
    with db() as con:
        run = con.execute(
            "SELECT id,status FROM scan_runs WHERE id=? AND user_id=?",
            (run_id, user_id),
        ).fetchone()
        if not run:
            return jsonify({"error": "Scan job not found"}), 404
        if run["status"] in ("completed", "failed"):
            return jsonify({"ok": True, "status": run["status"], "duplicate": True})
        if status_value != "completed":
            error_message = str(payload.get("error") or "GitHub Actions scanner failed")[-2500:]
            with db() as con2:
                con2.execute(
                    "UPDATE scan_runs SET status='failed',finished_at=?,error=? "
                    "WHERE id=? AND user_id=? AND status IN ('dispatched','running')",
                    (utc_now(), error_message, run_id, user_id),
                )
            return jsonify({"ok": True, "status": "failed"})

    results_csv = payload.get("results_csv")
    learning_payload = payload.get("learning_payload")
    base_learning_payload = payload.get("base_learning_payload")
    if not isinstance(results_csv, str) or not isinstance(learning_payload, str) or not isinstance(base_learning_payload, str):
        return jsonify({"error": "Scan callback is missing result or learning data"}), 400
    try:
        json.loads(learning_payload)
        json.loads(base_learning_payload)
        with tempfile.TemporaryDirectory(prefix=f"crypto-gh-scan-{run_id}-") as temp_name:
            temp_dir = Path(temp_name)
            learning_path = temp_dir / "ai_learning.json"
            results_path = temp_dir / "crypto_scan_results.csv"
            learning_path.write_text(learning_payload, encoding="utf-8")
            results_path.write_text(results_csv, encoding="utf-8")
            persist_shared_learning_and_observations(
                run_id, user_id, learning_path, results_path, base_learning_payload
            )
            results = parse_top_coins(results_path)
        with db() as con:
            con.execute(
                "UPDATE scan_runs SET status='completed',finished_at=?,top_coins_json=?,error=NULL "
                "WHERE id=? AND user_id=? AND status IN ('dispatched','running')",
                (utc_now(), json.dumps(results), run_id, user_id),
            )
        app.logger.info("GitHub Actions scan completed run_id=%s results=%s", run_id, len(results))
        return jsonify({"ok": True, "status": "completed", "result_count": len(results)})
    except Exception as exc:
        app.logger.exception("Could not persist GitHub Actions scan run_id=%s", run_id)
        with db() as con:
            con.execute(
                "UPDATE scan_runs SET status='failed',finished_at=?,error=? WHERE id=? AND user_id=?",
                (utc_now(), str(exc)[-2500:], run_id, user_id),
            )
        return jsonify({"error": "Could not save scan results"}), 500


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
                    "error": last["error"],
                    "runner_mode": "github_actions" if github_actions_enabled() else "render_local"})


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
            active = con.execute(
                "SELECT id,status FROM scan_runs WHERE user_id=? AND status IN ('queued','claimed','dispatched','running') "
                "ORDER BY id DESC LIMIT 1",
                (user["id"],),
            ).fetchone()
            if active:
                return jsonify({"ok": True, "run_id": active["id"], "status": active["status"], "already_running": True})
            # Keep one current scan per account. Never delete an active run.
            con.execute("DELETE FROM scan_observations WHERE user_id=?", (user["id"],))
            con.execute("DELETE FROM scan_runs WHERE user_id=?", (user["id"],))
            cur = con.execute("INSERT INTO scan_runs(user_id,status,started_at,top_coins_json) VALUES(?,?,?,?) RETURNING id",
                              (user["id"], "queued", started, "[]"))
            run_id = cur.fetchone()["id"]
        # Every customer's request is queued; each run uses its own temporary
        # directory so scanner CSV and learning files do not collide.
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
