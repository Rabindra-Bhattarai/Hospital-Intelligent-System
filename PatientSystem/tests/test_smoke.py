"""Smoke tests — boot the real app via Streamlit's AppTest and check every
main screen renders without raising. Requires MongoDB reachable at MONGO_URI
and ADMIN_USERNAME/ADMIN_PASSWORD set (see .github/workflows/ci.yml).
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _fresh_app():
    return AppTest.from_file(APP_PATH, default_timeout=60)


def test_landing_page_renders():
    at = _fresh_app()
    at.run()
    assert not at.exception


def test_login_page_renders():
    at = _fresh_app()
    at.session_state["landing_seen"] = True
    at.run()
    assert not at.exception


@pytest.mark.skipif(
    not os.environ.get("ADMIN_PASSWORD"),
    reason="ADMIN_PASSWORD not set — admin account wasn't seeded",
)
def test_admin_dashboard_renders():
    at = _fresh_app()
    at.session_state["landing_seen"] = True
    at.session_state["logged_in"] = True
    at.session_state["role"] = "admin"
    at.session_state["username"] = os.environ.get("ADMIN_USERNAME", "bkt_his_admin")
    at.session_state["_sid"] = "test"
    at.session_state["_session_expires"] = 9999999999.0
    at.run()
    assert not at.exception
