"""Accounts and bearer tokens (scrypt password hashing, hashed tokens at rest)."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from typing import Any

from indiclearn.studio.db import Database, now

_USER_RE = re.compile(r"^[A-Za-z0-9_.\-@]{3,40}$")


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthService:
    def __init__(self, db: Database, token_ttl_days: int = 30) -> None:
        self.db = db
        self.ttl = token_ttl_days * 86400

    def _issue(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        t = now()
        self.db.run("INSERT INTO tokens(token_hash,user_id,created_at,expires_at) VALUES (?,?,?,?)", (_token_hash(token), user_id, t, t + self.ttl))
        return token

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "username": row["username"], "display_name": row["display_name"] or row["username"], "is_guest": bool(row["is_guest"]), "lang": row["lang"] or "en"}

    def register(self, username: str, password: str, display_name: str | None = None, lang: str = "en") -> dict[str, Any]:
        username = (username or "").strip()
        if not _USER_RE.match(username):
            raise AuthError("Username must be 3-40 characters: letters, digits, _ . - @")
        if len(password or "") < 6:
            raise AuthError("Password must be at least 6 characters.")
        if self.db.one("SELECT 1 FROM users WHERE lower(username)=lower(?)", (username,)):
            raise AuthError("That username is already taken.", 409)
        uid = uuid.uuid4().hex
        salt = secrets.token_bytes(16)
        self.db.run(
            "INSERT INTO users(id,username,display_name,pw_hash,salt,is_guest,lang,created_at) VALUES (?,?,?,?,?,0,?,?)",
            (uid, username, (display_name or username).strip()[:60], _hash_password(password, salt), salt.hex(), lang, now()),
        )
        row = self.db.one("SELECT * FROM users WHERE id=?", (uid,))
        return {"token": self._issue(uid), "user": self._public(row)}

    def login(self, username: str, password: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM users WHERE lower(username)=lower(?) AND is_guest=0", ((username or "").strip(),))
        if row is None or not row["pw_hash"]:
            raise AuthError("Wrong username or password.", 401)
        calc = _hash_password(password or "", bytes.fromhex(row["salt"]))
        if not hmac.compare_digest(calc, row["pw_hash"]):
            raise AuthError("Wrong username or password.", 401)
        return {"token": self._issue(row["id"]), "user": self._public(row)}

    def guest(self, lang: str = "en") -> dict[str, Any]:
        uid = uuid.uuid4().hex
        uname = f"guest-{uid[:8]}"
        self.db.run(
            "INSERT INTO users(id,username,display_name,pw_hash,salt,is_guest,lang,created_at) VALUES (?,?,?,?,?,1,?,?)",
            (uid, uname, "Guest student", None, None, lang, now()),
        )
        row = self.db.one("SELECT * FROM users WHERE id=?", (uid,))
        return {"token": self._issue(uid), "user": self._public(row)}

    def user_for_token(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        row = self.db.one(
            "SELECT u.* FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token_hash=? AND t.expires_at>?",
            (_token_hash(token), now()),
        )
        return self._public(row) if row else None

    def logout(self, token: str) -> None:
        self.db.run("DELETE FROM tokens WHERE token_hash=?", (_token_hash(token),))

    def set_lang(self, user_id: str, lang: str) -> None:
        self.db.run("UPDATE users SET lang=? WHERE id=?", (lang, user_id))
