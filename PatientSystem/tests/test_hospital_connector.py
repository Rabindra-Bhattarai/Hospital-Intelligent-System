"""Unit tests for src/hospital_connector.py.

Runs against a real MongoDB instance, matching this project's existing
testing convention (see test_smoke.py) of exercising a real service rather
than mocking it. Uses a dedicated `hospital_intelligence_test` database so it
never touches real application data, and drops its collections after every
test. Skipped automatically if no MongoDB is reachable at MONGO_URI.
"""
import os
import sys

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config

TEST_DB_NAME = "hospital_intelligence_test"


def _mongo_reachable() -> bool:
    try:
        MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=1000).admin.command("ping")
        return True
    except ServerSelectionTimeoutError:
        return False


pytestmark = pytest.mark.skipif(
    not _mongo_reachable(),
    reason=f"No MongoDB reachable at {config.MONGO_URI}",
)


@pytest.fixture(autouse=True)
def _use_test_database(monkeypatch):
    """Point every test at an isolated test database and wipe it after."""
    monkeypatch.setattr(config, "MONGO_DB", TEST_DB_NAME)
    yield
    MongoClient(config.MONGO_URI).drop_database(TEST_DB_NAME)


@pytest.fixture
def connector(monkeypatch):
    """Import hospital_connector fresh so its module-level state (none today,
    but future-proof) picks up the monkeypatched config, and create its
    indexes (including the active-booking-per-department uniqueness
    constraint) on the isolated test database."""
    import importlib
    import src.hospital_connector as hc
    importlib.reload(hc)
    hc.init_appointments_index()
    return hc


def _booking_payload(**overrides):
    data = dict(
        patient_username="test_patient",
        patient_id="P0001",
        full_name="Test Patient",
        phone="9800000000",
        hospital_id="H1",
        hospital_name="Test Hospital",
        department="Cardiology",
        severity="Moderate",
        admission_type="Elective",
        requested_date="2026-09-01",
        preferred_time="Morning",
        los_low=3.0, los_high=7.0,
        cost_low=10000.0, cost_high=20000.0,
    )
    data.update(overrides)
    return data


# ── submit_booking ────────────────────────────────────────────────────────────

def test_active_booking_uniqueness_is_enforced_by_the_database_itself(connector):
    """Regression test for the race condition: the find_one-then-insert check
    in submit_booking() is only a fast-path — this proves the underlying
    partial unique index rejects a second active booking for the same
    patient+department even when inserted directly, bypassing that check
    entirely (simulating two near-simultaneous requests racing past it)."""
    from pymongo.errors import DuplicateKeyError

    db = connector._db()
    base_doc = dict(
        patient_username="racer", department="Cardiology", status="pending",
        booking_ref="BKG900001", hospital_id="H1", hospital_name="Test",
        severity="Moderate", admission_type="Elective", requested_date="2026-09-01",
        preferred_time="Morning", notes="", los_low=1, los_high=2,
        cost_low=1, cost_high=2, admin_note="",
    )
    db.appointments.insert_one(dict(base_doc))

    with pytest.raises(DuplicateKeyError):
        db.appointments.insert_one(dict(base_doc, booking_ref="BKG900002"))


def test_submit_booking_surfaces_a_friendly_error_if_the_index_rejects_it(connector, monkeypatch):
    """If two requests somehow both pass the find_one pre-check (the race
    window), submit_booking must still fail cleanly via the DuplicateKeyError
    handler rather than raising or corrupting state. Simulated by making
    insert_one always raise, while find_one/counters still hit the real
    (otherwise-empty) test database — i.e. the pre-check correctly finds no
    duplicate, but the insert fails anyway, exactly like a genuine race."""
    from pymongo.errors import DuplicateKeyError

    real_db = connector._db()

    class _FailingAppointments:
        def __getattr__(self, name):
            return getattr(real_db.appointments, name)

        def insert_one(self, *a, **k):
            raise DuplicateKeyError("simulated race")

    class _FakeDB:
        appointments = _FailingAppointments()

        def __getattr__(self, name):
            return getattr(real_db, name)

    monkeypatch.setattr(connector, "_db", lambda: _FakeDB())

    ok, msg = connector.submit_booking(_booking_payload(department="Cardiology"))
    assert ok is False
    assert "already have an active booking" in msg


def test_submit_booking_succeeds_and_returns_a_booking_ref(connector):
    ok, ref = connector.submit_booking(_booking_payload())
    assert ok is True
    assert ref.startswith("BKG")


def test_submit_booking_refs_increment_sequentially(connector):
    _, ref1 = connector.submit_booking(_booking_payload(department="Cardiology"))
    _, ref2 = connector.submit_booking(_booking_payload(
        patient_username="another_patient", department="Neurology",
    ))
    n1 = int(ref1.replace("BKG", ""))
    n2 = int(ref2.replace("BKG", ""))
    assert n2 == n1 + 1


def test_submit_booking_blocks_a_second_active_booking_same_department(connector):
    ok1, ref1 = connector.submit_booking(_booking_payload(department="Cardiology"))
    assert ok1 is True

    ok2, msg2 = connector.submit_booking(_booking_payload(department="Cardiology"))
    assert ok2 is False
    assert ref1 in msg2  # error message should reference the existing booking


def test_submit_booking_allows_a_second_booking_different_department(connector):
    ok1, _ = connector.submit_booking(_booking_payload(department="Cardiology"))
    ok2, _ = connector.submit_booking(_booking_payload(department="Neurology"))
    assert ok1 is True
    assert ok2 is True


def test_submit_booking_allows_new_booking_after_prior_one_cancelled(connector):
    ok1, ref1 = connector.submit_booking(_booking_payload(department="Cardiology"))
    assert connector.update_booking_status(ref1, "cancelled") is True

    ok2, _ = connector.submit_booking(_booking_payload(department="Cardiology"))
    assert ok2 is True  # duplicate check only blocks pending/confirmed, not cancelled


# ── get_booking ───────────────────────────────────────────────────────────────

def test_get_booking_returns_the_submitted_data(connector):
    _, ref = connector.submit_booking(_booking_payload(full_name="Ram Shrestha"))
    doc = connector.get_booking(ref)
    assert doc["booking_ref"] == ref
    assert doc["full_name"] == "Ram Shrestha"
    assert doc["status"] == "pending"


def test_get_booking_returns_empty_dict_for_unknown_ref(connector):
    assert connector.get_booking("BKG999999") == {}


# ── update_booking_status ────────────────────────────────────────────────────

def test_update_booking_status_changes_status_and_note(connector):
    _, ref = connector.submit_booking(_booking_payload())
    ok = connector.update_booking_status(ref, "confirmed", admin_note="See you at 9am")
    assert ok is True

    doc = connector.get_booking(ref)
    assert doc["status"] == "confirmed"
    assert doc["admin_note"] == "See you at 9am"


# ── list_patient_bookings / list_all_bookings / count_pending_bookings ────────

def test_list_patient_bookings_only_returns_that_patients_bookings(connector):
    connector.submit_booking(_booking_payload(
        patient_username="alice", department="Cardiology"))
    connector.submit_booking(_booking_payload(
        patient_username="bob", department="Neurology"))

    alice_bookings = connector.list_patient_bookings("alice")
    assert len(alice_bookings) == 1
    assert alice_bookings[0]["patient_username"] == "alice"


def test_list_all_bookings_can_filter_by_status(connector):
    _, ref1 = connector.submit_booking(_booking_payload(department="Cardiology"))
    connector.submit_booking(_booking_payload(
        patient_username="patient2", department="Neurology"))
    connector.update_booking_status(ref1, "confirmed")

    pending = connector.list_all_bookings(status_filter="pending")
    confirmed = connector.list_all_bookings(status_filter="confirmed")
    assert len(pending) == 1
    assert len(confirmed) == 1
    assert confirmed[0]["booking_ref"] == ref1


def test_count_pending_bookings_reflects_only_pending(connector):
    _, ref1 = connector.submit_booking(_booking_payload(department="Cardiology"))
    connector.submit_booking(_booking_payload(
        patient_username="patient2", department="Neurology"))
    assert connector.count_pending_bookings() == 2

    connector.update_booking_status(ref1, "confirmed")
    assert connector.count_pending_bookings() == 1
