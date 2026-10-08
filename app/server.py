"""Local website: sign in, upload a skin-lesion photo, and see what all four trained models predict.

Runs on CPU by default, so no GPU, internet connection or API key is needed.

    python app/server.py            then open http://localhost:8000
    python app/server.py --gpu      use an NVIDIA GPU if one is available

Privacy: uploaded photos are only held in memory while the models look at them and are
discarded right after. They are never written to disk, never put in the database, never
logged and never sent anywhere. The database (app/data/users.db) holds only usernames,
password hashes and login sessions.
"""
import argparse
import json
import webbrowser
from http import cookies
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from auth import SESSION_DAYS, AuthError, UserStore
from predictor import CLASS_INFO, GUIDANCE, SERIOUS, URGENT_SIGNS, InvalidImage, Predictor

APP = Path(__file__).resolve().parent
MAX_UPLOAD = 20 * 1024 * 1024
MAX_JSON = 16 * 1024
CSRF_HEADER = "X-Requested-With"  # browsers can't add custom headers to cross-site requests

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
                               "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
}


class Handler(SimpleHTTPRequestHandler):
    predictor = None
    users = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP / "static"), **kwargs)

    # ---- plumbing -------------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # no request logs: nothing about users or uploads is recorded

    def end_headers(self):
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")  # always revalidate, so updates show up immediately
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def send_json(self, obj, status=200, cookie=None):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def session_token(self):
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return jar["session"].value if "session" in jar else None

    def current_user(self):
        return self.users.session_user(self.session_token())

    @staticmethod
    def session_cookie(token, max_age=SESSION_DAYS * 86400):
        return f"session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}"

    def read_body(self, limit):
        return self.body if 0 < len(self.body) <= limit else None

    def read_json(self):
        body = self.read_body(MAX_JSON)
        try:
            data = json.loads(body) if body else None
        except ValueError:
            data = None
        return data if isinstance(data, dict) else None

    # ---- routes ---------------------------------------------------------------
    def do_GET(self):
        if self.path == "/api/info":
            p = self.predictor
            return self.send_json({**p.meta, "device": "GPU" if p.device.type == "cuda" else "CPU",
                                   "classes": {c: {"name": n, "about": a, "serious": c in SERIOUS, "care": GUIDANCE[c]}
                                               for c, (n, a) in CLASS_INFO.items()},
                                   "urgent_signs": URGENT_SIGNS})
        if self.path == "/api/me":
            user = self.current_user()
            return self.send_json({"user": user[1] if user else None})
        if self.path.startswith("/api/"):
            return self.send_json({"error": "Not found."}, 404)
        return super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            self.close_connection = True
            return self.send_json({"error": "Please upload an image under 20 MB."}, 413)
        # Read the whole body before answering anything: replying early and closing a socket that
        # still has unread data makes Windows reset the connection instead of delivering the reply.
        self.body = self.rfile.read(length) if length else b""
        if self.headers.get(CSRF_HEADER) != "fetch":
            return self.send_json({"error": "Bad request."}, 403)
        routes = {
            "/api/signup": self.signup, "/api/login": self.login, "/api/logout": self.logout,
            "/api/delete-account": self.delete_account, "/api/predict": self.predict,
        }
        handler = routes.get(self.path)
        if handler is None:
            return self.send_json({"error": "Not found."}, 404)
        return handler()

    def signup(self):
        data = self.read_json()
        if data is None:
            return self.send_json({"error": "Bad request."}, 400)
        try:
            user_id = self.users.create_user(data.get("username"), data.get("password"))
        except AuthError as e:
            return self.send_json({"error": str(e)}, 400)
        token = self.users.create_session(user_id)
        return self.send_json({"user": data["username"].strip()}, cookie=self.session_cookie(token))

    def login(self):
        data = self.read_json()
        if data is None:
            return self.send_json({"error": "Bad request."}, 400)
        try:
            user_id = self.users.authenticate(data.get("username"), data.get("password"))
        except AuthError as e:
            return self.send_json({"error": str(e)}, 401)
        token = self.users.create_session(user_id)
        username = self.users.session_user(token)[1]
        return self.send_json({"user": username}, cookie=self.session_cookie(token))

    def logout(self):
        self.users.end_session(self.session_token())
        return self.send_json({"ok": True}, cookie=self.session_cookie("", max_age=0))

    def delete_account(self):
        user = self.current_user()
        data = self.read_json()
        if user is None:
            return self.send_json({"error": "Please sign in."}, 401)
        if data is None:
            return self.send_json({"error": "Bad request."}, 400)
        try:
            self.users.authenticate(user[1], data.get("password"))
        except AuthError as e:
            msg = "That password isn't right." if str(e).startswith("Wrong") else str(e)
            return self.send_json({"error": msg}, 401)
        self.users.delete_user(user[0])
        return self.send_json({"ok": True}, cookie=self.session_cookie("", max_age=0))

    def predict(self):
        if self.current_user() is None:
            return self.send_json({"error": "Please sign in to analyse images."}, 401)
        image = self.read_body(MAX_UPLOAD)
        if image is None:
            return self.send_json({"error": "Please upload an image under 20 MB."}, 400)
        try:
            results = self.predictor.predict(image)
        except InvalidImage:
            return self.send_json({"error": "That file isn't an image this app can read. Try a JPG or PNG."}, 400)
        finally:
            del image  # the photo exists only in memory, only for this request
            self.body = b""
        return self.send_json({"results": results})


def make_server(predictor, users, port=8000, host="127.0.0.1"):
    Handler.predictor, Handler.users = predictor, users
    return ThreadingHTTPServer((host, port), Handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu", action="store_true", help="use an NVIDIA GPU if available (default: CPU)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--db", type=Path, default=APP / "data" / "users.db", help="accounts database file")
    args = ap.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device.type.upper()}...")
    server = make_server(Predictor(device), UserStore(args.db), args.port)
    url = f"http://localhost:{args.port}"
    print(f"Ready: {url}   (press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
