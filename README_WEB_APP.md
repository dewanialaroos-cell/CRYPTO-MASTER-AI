# Crypto Master AI Web Application

Dark/gold web dashboard for the existing Crypto Master AI scanner.

## Features
- Email/password signup and login (passwords stored as salted hashes).
- Authenticated dashboard and logout.
- Run button executes the existing `Scanner.py` server-side.
- Ranked LONG/SHORT candidates with price, strength, confidence, stop-loss and take-profit.
- One scanner process at a time because the scanner shares its learning and CSV files.
- No exchange orders are placed by the web app.

## Deploy
This is a long-running Python web service, not a static website or GitHub Pages app. Deploy to a persistent Python host that supports background subprocesses and persistent disk (for SQLite and scanner state). Configure:
- `APP_SECRET_KEY`: a long random secret (required in production)
- `COOKIE_SECURE=1` behind HTTPS
- Optional existing scanner variables such as `CMC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Persistent storage for `app_users.sqlite3`, `ai_learning.json`, and scan output.

Build command: `pip install -r requirements.txt`
Start command: `gunicorn --workers 1 --threads 4 --timeout 1000 --bind 0.0.0.0:$PORT app:app`

## Security / production notes
- Use HTTPS and set `APP_SECRET_KEY` in the hosting provider; do not rely on the development fallback.
- This initial version is a single-instance application. It serializes scans because `Scanner.py` updates shared files.
- Add email verification, password reset, and per-account rate limiting before opening signup publicly.
- Scan outputs are saved per account in the app database. Scanner-generated global learning state is shared by the engine.
- Trading signals are informational, not guaranteed. This app does not place trades.
