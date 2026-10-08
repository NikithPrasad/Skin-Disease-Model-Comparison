"""Tests for the website: accounts, sessions, request protection, and the guarantee that
uploaded photos are never written anywhere.

Run:  python -m unittest discover -s tests -v
"""
import http.client
import io
import json
import os
import secrets
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import auth  # noqa: E402
from auth import AuthError, UserStore  # noqa: E402
from server import make_server  # noqa: E402


def png_with_marker(marker):
    """A small PNG whose bytes contain `marker`, so we can search the disk for any copy of it."""
    info = PngImagePlugin.PngInfo()
    info.add_text("marker", marker)
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (180, 120, 100)).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


class FakePredictor:
    """Stands in for the real models so server tests run in milliseconds."""
    device = type("D", (), {"type": "cpu"})()
    meta = {"models": [], "seeds": 1}

    def predict(self, image_bytes):
        from predictor import preprocess  # still exercises real decoding / validation
        preprocess(image_bytes)
        return [{"model": "fake", "name": "Fake", "ms": 0.1, "top": "nv", "probs": [{"code": "nv", "p": 1.0}]}]


class UserStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = UserStore(Path(self.tmp.name) / "users.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_password_is_hashed_not_stored(self):
        self.store.create_user("alice", "correct horse")
        with closing(sqlite3.connect(self.store.db_path)) as db:
            stored = db.execute("SELECT password_hash FROM users").fetchone()[0]
        self.assertNotIn("correct horse", stored)
        self.assertTrue(auth.verify_password("correct horse", stored))
        self.assertFalse(auth.verify_password("wrong horse", stored))

    def test_same_password_gets_different_salt(self):
        self.assertNotEqual(auth.hash_password("samepassword"), auth.hash_password("samepassword"))

    def test_login(self):
        uid = self.store.create_user("Alice", "correct horse")
        self.assertEqual(self.store.authenticate("alice", "correct horse"), uid)  # case-insensitive username
        with self.assertRaises(AuthError):
            self.store.authenticate("alice", "wrong")
        with self.assertRaises(AuthError):
            self.store.authenticate("nobody", "correct horse")

    def test_validation(self):
        for name, pw in [("ab", "longenough"), ("bad name", "longenough"), ("x" * 33, "longenough"),
                         ("alice", "short"), ("alice", ""), ("alice", "alice"), (None, "longenough")]:
            with self.subTest(name=name, pw=pw), self.assertRaises(AuthError):
                self.store.create_user(name, pw)

    def test_duplicate_username_case_insensitive(self):
        self.store.create_user("alice", "password1")
        with self.assertRaises(AuthError):
            self.store.create_user("ALICE", "password2")

    def test_lockout_after_repeated_failures(self):
        self.store.create_user("alice", "correct horse")
        for _ in range(auth.MAX_FAILS):
            with self.assertRaises(AuthError):
                self.store.authenticate("alice", "wrong")
        with self.assertRaisesRegex(AuthError, "Too many"):
            self.store.authenticate("alice", "correct horse")  # even the right password is refused for now
        with mock.patch("auth.time.time", return_value=time.time() + auth.WINDOW + 1):
            self.assertTrue(self.store.authenticate("alice", "correct horse"))

    def test_sessions(self):
        uid = self.store.create_user("alice", "correct horse")
        token = self.store.create_session(uid)
        self.assertEqual(self.store.session_user(token), (uid, "alice"))
        with closing(sqlite3.connect(self.store.db_path)) as db:
            stored = db.execute("SELECT token_hash FROM sessions").fetchone()[0]
        self.assertNotEqual(stored, token)  # only a hash of the token is stored
        self.store.end_session(token)
        self.assertIsNone(self.store.session_user(token))
        self.assertIsNone(self.store.session_user("made-up-token"))
        self.assertIsNone(self.store.session_user(None))

    def test_expired_session(self):
        uid = self.store.create_user("alice", "correct horse")
        token = self.store.create_session(uid)
        with mock.patch("auth.time.time", return_value=time.time() + (auth.SESSION_DAYS + 1) * 86400):
            self.assertIsNone(self.store.session_user(token))

    def test_delete_user_removes_sessions(self):
        uid = self.store.create_user("alice", "correct horse")
        token = self.store.create_session(uid)
        self.store.delete_user(uid)
        self.assertIsNone(self.store.session_user(token))
        with closing(sqlite3.connect(self.store.db_path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)

    def test_database_has_no_place_for_images(self):
        with closing(sqlite3.connect(self.store.db_path)) as db:
            cols = {(t, c[1], c[2]) for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    for c in db.execute(f"PRAGMA table_info({t})")}
        self.assertFalse([c for c in cols if c[2].upper() == "BLOB"], "no BLOB columns allowed")
        self.assertEqual({t for t, _, _ in cols}, {"users", "sessions"})


class ServerTests(unittest.TestCase):
    predictor = FakePredictor()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = UserStore(Path(self.tmp.name) / "users.db")
        self.server = make_server(self.predictor, self.store, port=0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None, cookie=None, csrf=True):
        h = dict(headers or {})
        if csrf and method == "POST":
            h["X-Requested-With"] = "fetch"
        if cookie:
            h["Cookie"] = cookie
        if isinstance(body, dict):
            body, h["Content-Type"] = json.dumps(body).encode(), "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, body=body, headers=h)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        try:
            data = json.loads(data)
        except ValueError:
            pass
        return res, data

    def signup(self, name="alice", pw="correct horse"):
        res, _ = self.request("POST", "/api/signup", {"username": name, "password": pw})
        self.assertEqual(res.status, 200)
        return res.getheader("Set-Cookie").split(";")[0]

    def test_page_has_security_headers(self):
        res, body = self.request("GET", "/")
        self.assertEqual(res.status, 200)
        self.assertIn(b"Skin Lesion Classifier", body)
        self.assertIn("script-src 'self'", res.getheader("Content-Security-Policy"))
        self.assertEqual(res.getheader("X-Content-Type-Options"), "nosniff")

    def test_no_directory_listing_or_path_escape(self):
        self.assertEqual(self.request("GET", "/examples/")[0].status, 404)
        res, body = self.request("GET", "/../server.py")
        self.assertFalse(isinstance(body, bytes) and b"make_server" in body)

    def test_signup_cookie_is_protected(self):
        res, data = self.request("POST", "/api/signup", {"username": "alice", "password": "correct horse"})
        self.assertEqual(data, {"user": "alice"})
        cookie = res.getheader("Set-Cookie")
        for flag in ("HttpOnly", "SameSite=Strict", "Path=/"):
            self.assertIn(flag, cookie)

    def test_me_login_logout(self):
        cookie = self.signup()
        self.assertEqual(self.request("GET", "/api/me", cookie=cookie)[1], {"user": "alice"})
        self.request("POST", "/api/logout", {}, cookie=cookie)
        self.assertEqual(self.request("GET", "/api/me", cookie=cookie)[1], {"user": None})  # old cookie is dead
        res, data = self.request("POST", "/api/login", {"username": "ALICE", "password": "correct horse"})
        self.assertEqual((res.status, data["user"]), (200, "alice"))
        res, data = self.request("POST", "/api/login", {"username": "alice", "password": "nope"})
        self.assertEqual(res.status, 401)
        self.assertNotIn("Set-Cookie", dict(res.getheaders()))

    def test_post_without_csrf_header_is_rejected(self):
        res, _ = self.request("POST", "/api/signup", {"username": "alice", "password": "correct horse"}, csrf=False)
        self.assertEqual(res.status, 403)

    def test_predict_requires_login(self):
        res, data = self.request("POST", "/api/predict", png_with_marker("x"))
        self.assertEqual(res.status, 401)

    def test_predict(self):
        cookie = self.signup()
        res, data = self.request("POST", "/api/predict", png_with_marker("x"), cookie=cookie)
        self.assertEqual(res.status, 200)
        self.assertEqual(res.getheader("Cache-Control"), "no-store")
        self.assertEqual(data["results"][0]["top"], "nv")

    def test_bad_uploads(self):
        cookie = self.signup()
        self.assertEqual(self.request("POST", "/api/predict", b"not an image", cookie=cookie)[0].status, 400)
        self.assertEqual(self.request("POST", "/api/predict", b"", cookie=cookie)[0].status, 400)
        self.assertEqual(self.request("POST", "/api/signup", b"{not json", cookie=cookie)[0].status, 400)

    def test_delete_account(self):
        cookie = self.signup()
        self.assertEqual(self.request("POST", "/api/delete-account", {"password": "wrong"}, cookie=cookie)[0].status, 401)
        self.assertEqual(self.request("POST", "/api/delete-account", {"password": "correct horse"}, cookie=cookie)[0].status, 200)
        self.assertEqual(self.request("GET", "/api/me", cookie=cookie)[1], {"user": None})
        self.assertEqual(self.request("POST", "/api/login", {"username": "alice", "password": "correct horse"})[0].status, 401)

    def test_uploaded_photo_is_never_written_to_disk(self):
        cookie = self.signup()
        marker = "PRIVACY-" + secrets.token_hex(16)
        image = png_with_marker(marker)
        watched = [ROOT / "app", Path(self.tmp.name), Path(tempfile.gettempdir())]
        start = time.time() - 1
        res, _ = self.request("POST", "/api/predict", image, cookie=cookie)
        self.assertEqual(res.status, 200)
        leaks = []
        for base in watched:
            for dirpath, _, files in os.walk(base):
                for f in files:
                    p = Path(dirpath) / f
                    try:
                        if p.stat().st_mtime >= start and p.stat().st_size < 200_000_000 and marker.encode() in p.read_bytes():
                            leaks.append(p)
                    except OSError:
                        continue
        self.assertEqual(leaks, [], "uploaded photo found on disk")


class GuidanceTests(unittest.TestCase):
    def test_every_lesion_type_has_advice(self):
        from predictor import CLASSES, GUIDANCE, URGENT_SIGNS
        self.assertEqual(set(GUIDANCE), set(CLASSES))
        for code, g in GUIDANCE.items():
            self.assertIn(g["level"], {"urgent", "doctor", "selfcare"}, code)
            self.assertTrue(g["headline"] and g["looks"] and g["treatment"] and len(g["steps"]) >= 2, code)
        self.assertEqual(GUIDANCE["mel"]["level"], "urgent")  # melanoma always means see a doctor
        self.assertTrue(URGENT_SIGNS)


@unittest.skipUnless((ROOT / "app" / "models" / "meta.json").exists(), "trained models not exported yet")
class RealModelTests(ServerTests):
    """Runs the same server tests against the real trained models."""

    @classmethod
    def setUpClass(cls):
        from predictor import Predictor
        cls.predictor = Predictor()

    def test_predict(self):
        cookie = self.signup()
        image = (ROOT / "app" / "static" / "examples" / "nv.jpg").read_bytes()
        res, data = self.request("POST", "/api/predict", image, cookie=cookie)
        self.assertEqual(res.status, 200)
        self.assertEqual(len(data["results"]), 4)
        for r in data["results"]:
            self.assertAlmostEqual(sum(p["p"] for p in r["probs"]), 1.0, places=2)
            self.assertTrue(r["attention"].startswith("data:image/jpeg;base64,"))  # Grad-CAM image, in memory only


if __name__ == "__main__":
    unittest.main()
