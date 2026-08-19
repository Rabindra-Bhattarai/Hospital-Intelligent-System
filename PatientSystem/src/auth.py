"""Authentication + user profile layer — MongoDB backed."""
import hashlib, hmac, os, secrets, base64, re, time
import datetime, urllib.parse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError


# ── DB connection helper ────────────────────────────────────────────────────

def _db():
    client = MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB]


# ── Password validation ─────────────────────────────────────────────────────

def validate_password(password: str) -> tuple:
    """
    Returns (is_valid: bool, errors: list[str]).
    Rules: 8+ chars, uppercase, lowercase, digit, special character.
    """
    errors = []
    if len(password) < 8:
        errors.append("At least 8 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("At least one uppercase letter (A-Z)")
    if not re.search(r"[a-z]", password):
        errors.append("At least one lowercase letter (a-z)")
    if not re.search(r"\d", password):
        errors.append("At least one number (0-9)")
    if not re.search(r"[!@#$%^&*()\-_=+\[\]{};:'\",.<>?/\\|`~]", password):
        errors.append("At least one special character  (!  @  #  $  %  ^  &  *)")
    return (len(errors) == 0, errors)


# ── Hashing ─────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return salt, digest


def _verify_hash(password: str, salt: str, stored_hash: str) -> bool:
    _, digest = _hash_password(password, salt)
    return hmac.compare_digest(digest, stored_hash)


# ── DB initialisation ──────────────────────────────────────────────────────

def _next_patient_id() -> str:
    """Atomically increment the patient counter and return the next PAT###### ID."""
    db     = _db()
    result = db.counters.find_one_and_update(
        {"_id": "patient_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    return f"PAT{result['seq']:06d}"


def init_db() -> None:
    os.makedirs(config.AVATARS_DIR, exist_ok=True)
    db = _db()

    # Indexes
    db.users.create_index([("username", ASCENDING)], unique=True)
    db.users.create_index([("patient_id", ASCENDING)], sparse=True)
    db.profiles.create_index([("username", ASCENDING)], unique=True)
    db.sessions.create_index([("token", ASCENDING)], unique=True)
    db.sessions.create_index([("last_active", ASCENDING)])
    db.estimate_requests.create_index([("username", ASCENDING)])
    db.estimate_requests.create_index([("recorded_at", ASCENDING)])
    db.chat_messages.create_index([("patient_username", ASCENDING)])
    db.chat_messages.create_index([("sent_at", ASCENDING)])
    

    from src.hospital_connector import init_appointments_index
    init_appointments_index()

    # Seed default admin if absent — requires ADMIN_PASSWORD to be set via
    # env/.env so no account is ever created with a blank password.
    if config.DEFAULT_ADMIN_PASSWORD and db.users.find_one({"username": config.DEFAULT_ADMIN_USERNAME}) is None:
        salt, phash = _hash_password(config.DEFAULT_ADMIN_PASSWORD)
        db.users.insert_one({
            "username":      config.DEFAULT_ADMIN_USERNAME,
            "salt":          salt,
            "password_hash": phash,
            "role":          config.DEFAULT_ADMIN_ROLE,
            "created_at":    datetime.datetime.utcnow(),
        })


# ── Auth API ───────────────────────────────────────────────────────────────

def register_user(username: str, password: str, full_name: str = "", role: str = "patient",
                  dob: str = "", gender: str = "", phone: str = "", email: str = "",
                  district: str = "") -> tuple:
    role = "patient"  # patients cannot self-register as admin
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    ok, errs = validate_password(password)
    if not ok:
        return False, "Weak password — " + "; ".join(errs) + "."

    salt, phash = _hash_password(password)
    try:
        db         = _db()
        patient_id = _next_patient_id()
        db.users.insert_one({
            "username":      username.strip(),
            "salt":          salt,
            "password_hash": phash,
            "role":          role,
            "patient_id":    patient_id,
            "created_at":    datetime.datetime.utcnow(),
        })
        db.profiles.insert_one({
            "username":           username.strip(),
            "full_name":          full_name.strip(),
            "dob":                dob.strip(),
            "gender":             gender.strip() if gender.strip() != "Prefer not to say" else "",
            "district":           district.strip(),
            "phone":              phone.strip(),
            "email":              email.strip(),
            "blood_type":         "Unknown",
            "allergies":          "",
            "occupation":         "",
            "chronic_conditions": "",
            "ec_name":            "",
            "ec_relationship":    "",
            "ec_phone":           "",
            "pref_hospital":      "Any hospital",
            "avatar_filename":    "",
            "updated_at":         datetime.datetime.utcnow(),
        })
        return True, "Account created successfully."
    except DuplicateKeyError:
        return False, "Username already taken. Please choose another."
    except Exception as e:
        return False, f"Registration error: {e}"


def verify_user(username: str, password: str):
    """Return role string on success, None on failure."""
    try:
        db  = _db()
        doc = db.users.find_one({"username": username.strip()})
        if doc is None:
            return None
        return doc["role"] if _verify_hash(password, doc["salt"], doc["password_hash"]) else None
    except Exception:
        return None


def change_password(username: str, old_password: str,
                    new_password: str) -> tuple:
    if new_password == old_password:
        return False, "New password must differ from the current one."
    ok, errs = validate_password(new_password)
    if not ok:
        return False, "Weak password — " + "; ".join(errs) + "."
    if verify_user(username, old_password) is None:
        return False, "Current password is incorrect."
    salt, phash = _hash_password(new_password)
    try:
        db = _db()
        db.users.update_one(
            {"username": username},
            {"$set": {"salt": salt, "password_hash": phash}},
        )
        return True, "Password changed successfully."
    except Exception as e:
        return False, f"Error: {e}"


def get_user_created_at(username: str) -> str:
    try:
        db  = _db()
        doc = db.users.find_one({"username": username}, {"created_at": 1})
        if doc and "created_at" in doc:
            v = doc["created_at"]
            return v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)[:10]
        return ""
    except Exception:
        return ""


# ── Profile API ─────────────────────────────────────────────────────────────

_PROFILE_DEFAULTS = {
    "full_name": "", "dob": "", "gender": "", "district": "",
    "phone": "", "email": "", "blood_type": "Unknown",
    "allergies": "", "occupation": "", "chronic_conditions": "",
    "ec_name": "", "ec_relationship": "", "ec_phone": "",
    "pref_hospital": "Any hospital", "avatar_filename": "",
}

_PROFILE_FIELDS = list(_PROFILE_DEFAULTS.keys())


def get_profile(username: str) -> dict:
    try:
        db  = _db()
        doc = db.profiles.find_one({"username": username})
        if doc is None:
            defaults = {"username": username, **_PROFILE_DEFAULTS,
                        "updated_at": datetime.datetime.utcnow()}
            db.profiles.insert_one(defaults)
            doc = defaults
        doc.pop("_id", None)
        return doc
    except Exception:
        return {}


def save_profile(username: str, data: dict) -> tuple:
    filtered = {k: v for k, v in data.items() if k in _PROFILE_FIELDS}
    if not filtered:
        return False, "Nothing to save."
    filtered["updated_at"] = datetime.datetime.utcnow()
    try:
        db = _db()
        db.profiles.update_one(
            {"username": username},
            {"$set": filtered},
            upsert=True,
        )
        return True, "Profile saved."
    except Exception as e:
        return False, f"Save error: {e}"


# ── Avatar API ───────────────────────────────────────────────────────────────

_ALLOWED_AVATAR_EXTS = {"png", "jpg", "jpeg"}


def save_avatar(username: str, image_bytes: bytes, ext: str = "png") -> tuple:
    # ext comes from the client-supplied upload filename — never trust it
    # blindly. A crafted extension (or username) containing "/" or ".."
    # could otherwise be used to write outside AVATARS_DIR.
    ext = str(ext).strip().lower()
    if ext not in _ALLOWED_AVATAR_EXTS:
        return False, "Unsupported file type — only JPG or PNG avatars are allowed."
    try:
        os.makedirs(config.AVATARS_DIR, exist_ok=True)
        avatars_root = os.path.realpath(config.AVATARS_DIR)
        filename     = f"{username}.{ext}"
        path         = os.path.realpath(os.path.join(avatars_root, filename))
        if os.path.commonpath([avatars_root, path]) != avatars_root:
            return False, "Invalid filename."
        for f in os.listdir(config.AVATARS_DIR):
            if f.startswith(f"{username}."):
                try:
                    os.remove(os.path.join(config.AVATARS_DIR, f))
                except Exception:
                    pass
        with open(path, "wb") as fh:
            fh.write(image_bytes)
        db = _db()
        db.profiles.update_one(
            {"username": username},
            {"$set": {"avatar_filename": filename, "updated_at": datetime.datetime.utcnow()}},
            upsert=True,
        )
        return True, filename
    except Exception as e:
        return False, str(e)


def _get_avatar_filename(username: str) -> str:
    try:
        db  = _db()
        doc = db.profiles.find_one({"username": username}, {"avatar_filename": 1})
        return (doc or {}).get("avatar_filename", "")
    except Exception:
        return ""


def get_avatar_b64(username: str) -> str:
    try:
        filename = _get_avatar_filename(username)
        if filename:
            path = os.path.join(config.AVATARS_DIR, filename)
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    return base64.b64encode(fh.read()).decode()
    except Exception:
        pass
    return ""


def get_avatar_thumb_b64(username: str, size: int = 96) -> str:
    try:
        filename = _get_avatar_filename(username)
        if not filename:
            return ""
        path = os.path.join(config.AVATARS_DIR, filename)
        if not os.path.exists(path):
            return ""
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            with open(path, "rb") as fh:
                return base64.b64encode(fh.read()).decode()
    except Exception:
        return ""


def get_avatar_url(username: str) -> str:
    try:
        filename = _get_avatar_filename(username)
        if filename:
            path = os.path.join(config.AVATARS_DIR, filename)
            if os.path.exists(path):
                mtime = int(os.path.getmtime(path))
                return f"/app/static/avatars/{filename}?v={mtime}"
    except Exception:
        pass
    return ""


# ── Estimate requests ────────────────────────────────────────────────────────

def save_estimate_request(data: dict) -> bool:
    """Persist one consent-given estimate request for analysis."""
    try:
        db = _db()
        repeat_count = db.estimate_requests.count_documents(
            {"username": data.get("username", "")}
        )
        db.estimate_requests.insert_one({
            "username":               data.get("username", ""),
            "recorded_at":            datetime.datetime.utcnow(),
            "age":                    data.get("age"),
            "gender":                 data.get("gender", ""),
            "severity":               data.get("severity", ""),
            "admission_type":         data.get("admission_type", ""),
            "department":             data.get("department", ""),
            "hospital_id":            data.get("hospital_id", ""),
            "has_chronic":            int(data.get("has_chronic", 0)),
            "has_insurance":          int(data.get("has_insurance", 0)),
            "visit_for":              data.get("visit_for", ""),
            "travel_dist":            data.get("travel_dist", ""),
            "los_low":                data.get("los_low"),
            "los_high":               data.get("los_high"),
            "cost_low":               data.get("cost_low"),
            "cost_high":              data.get("cost_high"),
            "district":               data.get("district", ""),
            "repeat_visit_count":     repeat_count,
            "hosp_occupancy_at_time": data.get("hosp_occupancy_at_time"),
            "num_chronic":            data.get("num_chronic"),
        })
        return True
    except Exception:
        return False


# ── Google SSO ──────────────────────────────────────────────────────────────

def get_google_auth_url(state: str = "sso") -> str:
    """Build the Google OAuth2 authorization URL to redirect the patient to."""
    params = {
        "client_id":     config.GOOGLE_CLIENT_ID,
        "redirect_uri":  config.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "online",
        "prompt":        "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_google_code(code: str) -> dict | None:
    """Exchange OAuth authorization code for Google user info dict."""
    try:
        import requests as _req
        token_resp = _req.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     config.GOOGLE_CLIENT_ID,
                "client_secret": config.GOOGLE_CLIENT_SECRET,
                "redirect_uri":  config.GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
            timeout=10,
        )
        if not token_resp.ok:
            return None
        access_token = token_resp.json().get("access_token")
        info_resp = _req.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        return info_resp.json() if info_resp.ok else None
    except Exception:
        return None


def get_or_create_sso_user(email: str, name: str, provider: str = "google") -> str:
    """Find or auto-create a patient account for an SSO user. Returns username."""
    username = f"{provider}_{email}"
    db = _db()
    if db.users.find_one({"username": username}) is None:
        patient_id = _next_patient_id()
        db.users.insert_one({
            "username":      username,
            "salt":          "",
            "password_hash": "",
            "role":          "patient",
            "patient_id":    patient_id,
            "sso_provider":  provider,
            "sso_email":     email,
            "created_at":    datetime.datetime.utcnow(),
        })
        db.profiles.insert_one({
            "username":   username,
            "full_name":  name,
            **{k: v for k, v in _PROFILE_DEFAULTS.items() if k != "full_name"},
            "updated_at": datetime.datetime.utcnow(),
        })
    return username


def get_patient_id(username: str) -> str:
    """Return the PAT###### ID for a patient, or '' if not found."""
    try:
        db  = _db()
        doc = db.users.find_one({"username": username}, {"patient_id": 1})
        return (doc or {}).get("patient_id", "")
    except Exception:
        return ""


# ── User existence check ─────────────────────────────────────────────────────

def user_exists(username: str) -> bool:
    try:
        db = _db()
        return db.users.find_one({"username": username.strip()}) is not None
    except Exception:
        return False


# ── Session API (refresh-proof login) ───────────────────────────────────────

def create_session(username: str, role: str) -> str:
    token = secrets.token_hex(32)
    db = _db()
    db.sessions.insert_one({
        "token":       token,
        "username":    username,
        "role":        role,
        "last_active": time.time(),
    })
    return token


def validate_session(token: str, timeout: float = 3600):
    """Return {username, role, last_active} if token is valid and not expired, else None."""
    if not token:
        return None
    try:
        db  = _db()
        doc = db.sessions.find_one({"token": token})
        if not doc:
            return None
        if time.time() - doc["last_active"] > timeout:
            revoke_session(token)
            return None
        return {"username": doc["username"], "role": doc["role"],
                "last_active": doc["last_active"]}
    except Exception:
        return None


def touch_session(token: str) -> None:
    try:
        db = _db()
        db.sessions.update_one({"token": token}, {"$set": {"last_active": time.time()}})
    except Exception:
        pass


def revoke_session(token: str) -> None:
    try:
        db = _db()
        db.sessions.delete_one({"token": token})
    except Exception:
        pass


# ── Admin: patient account management ───────────────────────────────────────

def list_patient_accounts() -> list:
    """Return all patient accounts enriched with basic profile data, newest first."""
    try:
        db    = _db()
        users = list(db.users.find(
            {"role": "patient"},
            {"username": 1, "patient_id": 1, "created_at": 1,
             "sso_provider": 1, "sso_email": 1, "_id": 0},
        ).sort("created_at", -1))
        for u in users:
            prof = db.profiles.find_one(
                {"username": u["username"]},
                {"full_name": 1, "district": 1, "gender": 1, "blood_type": 1,
                 "chronic_conditions": 1, "pref_hospital": 1, "_id": 0},
            ) or {}
            u.update({
                "full_name":          prof.get("full_name", ""),
                "district":           prof.get("district", ""),
                "gender":             prof.get("gender", ""),
                "blood_type":         prof.get("blood_type", "Unknown"),
                "chronic_conditions": prof.get("chronic_conditions", ""),
                "pref_hospital":      prof.get("pref_hospital", ""),
            })
        return users
    except Exception:
        return []


# ── Chat messaging ───────────────────────────────────────────────────────────

def send_chat_message(patient_username: str, sender: str, body: str) -> bool:
    """sender must be 'patient' or 'admin'."""
    if not body.strip():
        return False
    try:
        db = _db()
        db.chat_messages.insert_one({
            "patient_username": patient_username,
            "sender":           sender,
            "body":             body.strip(),
            "sent_at":          datetime.datetime.utcnow(),
            "read_by_admin":    sender == "admin",
            "read_by_patient":  sender == "patient",
        })
        return True
    except Exception:
        return False


def get_chat_messages(patient_username: str) -> list:
    """All messages in a patient thread, oldest first."""
    try:
        db = _db()
        return list(db.chat_messages.find(
            {"patient_username": patient_username}, {"_id": 0}
        ).sort("sent_at", 1))
    except Exception:
        return []


def get_chat_inbox() -> list:
    """Admin view: one entry per patient thread, sorted by latest message."""
    try:
        db = _db()
        pipeline = [
            {"$sort": {"sent_at": -1}},
            {"$group": {
                "_id":          "$patient_username",
                "latest_body":  {"$first": "$body"},
                "latest_at":    {"$first": "$sent_at"},
                "latest_sender":{"$first": "$sender"},
                "unread": {"$sum": {"$cond": [
                    {"$and": [
                        {"$eq": ["$sender", "patient"]},
                        {"$eq": ["$read_by_admin", False]},
                    ]}, 1, 0,
                ]}},
            }},
            {"$sort": {"latest_at": -1}},
        ]
        threads = list(db.chat_messages.aggregate(pipeline))
        for th in threads:
            uname = th["_id"]
            prof = db.profiles.find_one({"username": uname}, {"full_name": 1}) or {}
            user = db.users.find_one({"username": uname}, {"patient_id": 1}) or {}
            th["patient_username"] = uname
            th["full_name"]        = prof.get("full_name") or uname
            th["patient_id"]       = user.get("patient_id", "")
        return threads
    except Exception:
        return []


def mark_chat_read(patient_username: str, by_role: str) -> None:
    """Mark messages as read by 'admin' or 'patient'."""
    try:
        db = _db()
        if by_role == "admin":
            db.chat_messages.update_many(
                {"patient_username": patient_username, "sender": "patient"},
                {"$set": {"read_by_admin": True}},
            )
        else:
            db.chat_messages.update_many(
                {"patient_username": patient_username, "sender": "admin"},
                {"$set": {"read_by_patient": True}},
            )
    except Exception:
        pass


def count_unread_for_patient(patient_username: str) -> int:
    """Count admin messages not yet read by the patient."""
    try:
        db = _db()
        return db.chat_messages.count_documents({
            "patient_username": patient_username,
            "sender":           "admin",
            "read_by_patient":  False,
        })
    except Exception:
        return 0


def delete_patient_account(username: str) -> tuple:
    """Permanently delete a patient account and all associated data. Refuses admin accounts."""
    try:
        db  = _db()
        doc = db.users.find_one({"username": username})
        if not doc:
            return False, "Account not found."
        if doc.get("role") != "patient":
            return False, "Cannot delete admin accounts."
        db.users.delete_one({"username": username})
        db.profiles.delete_one({"username": username})
        db.sessions.delete_many({"username": username})
        db.estimate_requests.delete_many({"username": username})
        return True, f"Account '{username}' deleted successfully."
    except Exception as e:
        return False, f"Error: {e}"
