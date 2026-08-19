"""Unit tests for src/auth.py — password hashing correctness, and the avatar
upload path-traversal guard added after a security review found ext/username
could otherwise be used to write outside AVATARS_DIR.

save_avatar tests run against a real MongoDB (isolated test database, same
convention as test_hospital_connector.py) and a temp AVATARS_DIR, so they
never touch real application data or the real avatars folder.
"""
import os
import sys

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import src.auth as auth

TEST_DB_NAME = "hospital_intelligence_test"


def _mongo_reachable() -> bool:
    try:
        MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=1000).admin.command("ping")
        return True
    except ServerSelectionTimeoutError:
        return False


# ── _hash_password / _verify_hash — pure, no DB needed ─────────────────────────

def test_hash_password_generates_a_fresh_salt_each_time():
    salt1, digest1 = auth._hash_password("Sup3r$ecret")
    salt2, digest2 = auth._hash_password("Sup3r$ecret")
    assert salt1 != salt2
    assert digest1 != digest2  # different salt -> different digest, even for the same password


def test_verify_hash_accepts_correct_password():
    salt, digest = auth._hash_password("Correct$123")
    assert auth._verify_hash("Correct$123", salt, digest) is True


def test_verify_hash_rejects_wrong_password():
    salt, digest = auth._hash_password("Correct$123")
    assert auth._verify_hash("Wrong$1234", salt, digest) is False


def test_verify_hash_uses_constant_time_comparison(monkeypatch):
    # Regression check for the timing-safe fix: verification must go through
    # hmac.compare_digest rather than Python's short-circuiting `==`.
    calls = {}
    original_compare_digest = auth.hmac.compare_digest  # capture before patching

    def _tracking_compare_digest(a, b):
        calls["used"] = True
        return original_compare_digest(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", _tracking_compare_digest)
    salt, digest = auth._hash_password("Whatever$1")
    auth._verify_hash("Whatever$1", salt, digest)
    assert calls.get("used") is True


# ── save_avatar — path-traversal guard ─────────────────────────────────────────

pytestmark_avatar = pytest.mark.skipif(
    not _mongo_reachable(),
    reason=f"No MongoDB reachable at {config.MONGO_URI}",
)


@pytest.fixture
def avatar_env(monkeypatch, tmp_path):
    """Isolated AVATARS_DIR + isolated test Mongo database for every test."""
    monkeypatch.setattr(config, "AVATARS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MONGO_DB", TEST_DB_NAME)
    yield tmp_path
    MongoClient(config.MONGO_URI).drop_database(TEST_DB_NAME)


@pytestmark_avatar
def test_save_avatar_accepts_a_legitimate_upload(avatar_env):
    ok, filename = auth.save_avatar("normal_user", b"fake-image-bytes", ext="png")
    assert ok is True
    assert filename == "normal_user.png"
    assert os.path.exists(os.path.join(str(avatar_env), "normal_user.png"))


@pytestmark_avatar
def test_save_avatar_rejects_disallowed_extension(avatar_env):
    ok, msg = auth.save_avatar("normal_user", b"fake-bytes", ext="php")
    assert ok is False
    assert "Unsupported file type" in msg
    # nothing should have been written to disk
    assert os.listdir(str(avatar_env)) == []


@pytestmark_avatar
def test_save_avatar_rejects_path_traversal_via_extension(avatar_env):
    malicious_ext = "png/../../../../evil"
    ok, _ = auth.save_avatar("normal_user", b"fake-bytes", ext=malicious_ext)
    assert ok is False
    # nothing written anywhere, in particular not outside the temp avatars dir
    assert os.listdir(str(avatar_env)) == []


@pytestmark_avatar
def test_save_avatar_rejects_path_traversal_via_username(avatar_env):
    malicious_username = "../../../../evil"
    ok, msg = auth.save_avatar(malicious_username, b"fake-bytes", ext="png")
    assert ok is False
    assert msg == "Invalid filename."
    assert os.listdir(str(avatar_env)) == []


@pytestmark_avatar
def test_save_avatar_uppercase_extension_is_normalised(avatar_env):
    ok, filename = auth.save_avatar("normal_user", b"fake-bytes", ext="PNG")
    assert ok is True
    assert filename == "normal_user.png"
