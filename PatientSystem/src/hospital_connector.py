"""
Hospital Connector — abstraction layer between the app and the hospital backend.

TODAY  : all functions read/write MongoDB locally.
FUTURE : replace each function body to call a real hospital REST API.
         The app never needs to change — only this file.
"""
import datetime, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError


def _db():
    return MongoClient(config.MONGO_URI)[config.MONGO_DB]


def _next_booking_ref() -> str:
    db = _db()
    result = db.counters.find_one_and_update(
        {"_id": "booking_ref"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    return f"BKG{result['seq']:06d}"


def init_appointments_index() -> None:
    db = _db()
    db.appointments.create_index([("booking_ref", ASCENDING)], unique=True)
    db.appointments.create_index([("patient_username", ASCENDING)])
    db.appointments.create_index([("status", ASCENDING)])
    db.appointments.create_index([("requested_date", ASCENDING)])
    # Enforces "one active booking per patient per department" atomically at
    # the database level — the find_one-then-insert check in submit_booking()
    # below is a fast-path for a friendly error message, but on its own it's
    # a check-then-act race (two near-simultaneous submissions could both
    # pass it before either inserts). This partial unique index is the actual
    # guarantee: only pending/confirmed bookings are covered by it, so past
    # cancelled/completed bookings for the same patient+department don't
    # collide with it.
    db.appointments.create_index(
        [("patient_username", ASCENDING), ("department", ASCENDING)],
        unique=True,
        partialFilterExpression={"status": {"$in": ["pending", "confirmed"]}},
        name="uniq_active_booking_per_patient_dept",
    )


# ── FUTURE HOOK ─────────────────────────────────────────────────────────────
# To connect a real hospital API, replace the body of each function below.
# Keep the same parameters and return types — the app won't need to change.
# Example:
#   response = requests.post("https://his.hospital.gov.np/api/bookings", json=payload)
#   return True, response.json()["booking_ref"]
# ─────────────────────────────────────────────────────────────────────────────

def submit_booking(data: dict) -> tuple:
    """
    Create a new appointment request.
    Returns (success: bool, booking_ref_or_error: str).

    Required keys in data:
      patient_username, patient_id, full_name, phone,
      hospital_id, hospital_name, department,
      severity, admission_type,
      requested_date (str YYYY-MM-DD), preferred_time,
      los_low, los_high, cost_low, cost_high,
      notes (optional)
    """
    try:
        db = _db()
        # Enforce: one active booking per department per patient
        duplicate = db.appointments.find_one({
            "patient_username": data["patient_username"],
            "department":       data.get("department"),
            "status":           {"$in": ["pending", "confirmed"]},
        })
        if duplicate:
            return False, (
                f"You already have an active booking ({duplicate['booking_ref']}) "
                f"for {data.get('department')}. Cancel it first."
            )
        ref = _next_booking_ref()
        db.appointments.insert_one({
            "booking_ref":      ref,
            "patient_username": data["patient_username"],
            "patient_id":       data.get("patient_id", ""),
            "full_name":        data.get("full_name", ""),
            "phone":            data.get("phone", ""),
            "hospital_id":      data.get("hospital_id", ""),
            "hospital_name":    data.get("hospital_name", ""),
            "department":       data.get("department", ""),
            "severity":         data.get("severity", ""),
            "admission_type":   data.get("admission_type", ""),
            "requested_date":   data.get("requested_date", ""),
            "preferred_time":   data.get("preferred_time", "Morning"),
            "notes":            data.get("notes", ""),
            "los_low":          data.get("los_low"),
            "los_high":         data.get("los_high"),
            "cost_low":         data.get("cost_low"),
            "cost_high":        data.get("cost_high"),
            "status":           "pending",
            "admin_note":       "",
            "created_at":       datetime.datetime.utcnow(),
            "updated_at":       datetime.datetime.utcnow(),
        })
        return True, ref
    except DuplicateKeyError:
        # The find_one check above is a fast-path only — this is the actual
        # race-proof guarantee, hit when two near-simultaneous submissions
        # both passed that check before either had inserted.
        return False, (
            f"You already have an active booking for {data.get('department')}. "
            f"Cancel it first."
        )
    except Exception as e:
        return False, str(e)


def get_booking(booking_ref: str) -> dict:
    """Fetch a single booking by reference. Returns {} if not found."""
    try:
        db  = _db()
        doc = db.appointments.find_one({"booking_ref": booking_ref}, {"_id": 0})
        return doc or {}
    except Exception:
        return {}


def update_booking_status(booking_ref: str, status: str,
                          admin_note: str = "") -> bool:
    """
    Update booking status.
    Valid statuses: pending | confirmed | cancelled | completed | no_show
    """
    try:
        db = _db()
        db.appointments.update_one(
            {"booking_ref": booking_ref},
            {"$set": {
                "status":     status,
                "admin_note": admin_note,
                "updated_at": datetime.datetime.utcnow(),
            }},
        )
        return True
    except Exception:
        return False


def list_patient_bookings(patient_username: str) -> list:
    """All bookings for a patient, newest first."""
    try:
        db = _db()
        return list(db.appointments.find(
            {"patient_username": patient_username}, {"_id": 0}
        ).sort("created_at", -1))
    except Exception:
        return []


def list_all_bookings(status_filter: str = None) -> list:
    """Admin: all bookings, optionally filtered by status, newest first."""
    try:
        db    = _db()
        query = {"status": status_filter} if status_filter else {}
        return list(db.appointments.find(query, {"_id": 0}).sort("created_at", -1))
    except Exception:
        return []


def count_pending_bookings() -> int:
    """Admin: number of unactioned (pending) bookings."""
    try:
        return _db().appointments.count_documents({"status": "pending"})
    except Exception:
        return 0
