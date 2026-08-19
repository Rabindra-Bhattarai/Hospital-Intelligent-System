"""Hospital Intelligence System — Complete UI Redesign"""
import os, sys, time as _time, secrets, base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from src.auth         import (init_db, register_user, verify_user,
                               change_password, validate_password,
                               get_profile, save_profile,
                               save_avatar, get_avatar_thumb_b64,
                               get_user_created_at,
                               create_session, validate_session,
                               touch_session, revoke_session,
                               save_estimate_request,
                               get_google_auth_url, exchange_google_code,
                               get_or_create_sso_user, get_patient_id,
                               list_patient_accounts, delete_patient_account,
                               send_chat_message, get_chat_messages,
                               get_chat_inbox, mark_chat_read,
                               count_unread_for_patient)
from src.data_loader  import (load_admissions, load_patients, load_occupancy,
                               load_occupancy_with_hospital,
                               hospital_stats, departments_by_hospital,
                               hospitals_with_department,
                               merged_admissions_patients, kpi_summary,
                               thesis_model_metrics)
from src.predict      import predict_los, predict_cost, models_ready, model_metrics
from src.hospital_connector import (submit_booking, list_patient_bookings,
                                    list_all_bookings, update_booking_status,
                                    count_pending_bookings)
from src.decision_engine import occupancy_alert, overtime_alert, los_flag

st.set_page_config(
    page_title="Hospital Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session/role checks below hit the database before any page content exists —
# without this, the browser shows a blank tab for that entire round trip.
# Only worth the spinner when a real DB round-trip is actually about to
# happen (a token to validate/touch, or an OAuth code to exchange) — e.g.
# not on the rerun after Sign Out, which already cleared the session token
# and would otherwise show a needless full-screen flash.
_qp_boot = st.query_params
_needs_boot_spinner = bool(
    (st.session_state.get("logged_in") and st.session_state.get("_sid"))
    or (not st.session_state.get("logged_in") and _qp_boot.get("sid"))
    or (not st.session_state.get("logged_in") and "code" in _qp_boot)
)

_boot_gate = st.empty()
if _needs_boot_spinner:
    with _boot_gate.container():
        st.markdown("""
        <style>
          #MainMenu, header, footer { visibility: hidden; }
        </style>
        <div style="min-height:100dvh;display:flex;flex-direction:column;align-items:center;
          justify-content:center;gap:16px;">
          <div style="width:34px;height:34px;border-radius:50%;
            border:3px solid #DCEAE6;border-top-color:#0F766E;
            animation:bootSpin .8s linear infinite;"></div>
          <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
            color:#102A2A;font-size:.92rem;letter-spacing:-.01em;">Loading Hospital Intelligence System…</div>
        </div>
        <style>
          @keyframes bootSpin { to { transform:rotate(360deg); } }
          @media (prefers-reduced-motion:reduce) {
            [style*="bootSpin"] { animation:none !important; opacity:.6; }
          }
        </style>
        """, unsafe_allow_html=True)

init_db()

for k, v in [("logged_in", False), ("username", ""), ("role", ""),
             ("page", "dashboard"), ("profile_msg", None),
             ("login_view", "signin"), ("reg_success", ""), ("reg_step", 1),
             ("_sid", ""), ("_session_expires", 0.0), ("landing_seen", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# Links inside the landing component navigate the top-level window. Query
# parameters provide a reliable bridge from that iframe into Streamlit state.
_auth_route = st.query_params.get("auth", "")
if not st.session_state.logged_in and _auth_route in ("signin", "register"):
    st.session_state.landing_seen = True
    st.session_state.login_view = _auth_route
    st.session_state.reg_step = 1
    del st.query_params["auth"]

SESSION_TIMEOUT = 3600  # seconds of inactivity before auto-logout


def _restore_session():
    """On every page load: restore login from URL token, or touch active session."""
    if st.session_state.logged_in:
        sid = st.session_state.get("_sid", "")
        if sid:
            touch_session(sid)
            st.session_state._session_expires = _time.time() + SESSION_TIMEOUT
        return
    sid = st.query_params.get("sid", None)
    if not sid:
        return
    result = validate_session(sid, SESSION_TIMEOUT)
    if result:
        st.session_state.logged_in        = True
        st.session_state.username         = result["username"]
        st.session_state.role             = result["role"]
        st.session_state._sid             = sid
        st.session_state._session_expires = _time.time() + SESSION_TIMEOUT
        touch_session(sid)
    else:
        try:
            st.query_params.clear()
        except Exception:
            pass


def _save_session(username: str, role: str):
    """Create server-side token and persist it in the URL."""
    token = create_session(username, role)
    st.session_state._sid             = token
    st.session_state._session_expires = _time.time() + SESSION_TIMEOUT
    st.query_params["sid"] = token


def _clear_session():
    """Revoke token and clear URL param (logout)."""
    sid = st.session_state.get("_sid", "") or st.query_params.get("sid", "")
    if sid:
        revoke_session(sid)
    st.session_state._sid             = ""
    st.session_state._session_expires = 0.0
    try:
        st.query_params.clear()
    except Exception:
        pass


_restore_session()

# ── Google OAuth callback handler ────────────────────────────────────────────
if not st.session_state.logged_in:
    _qp = st.query_params
    if "code" in _qp:
        _info = exchange_google_code(_qp["code"])
        if _info and _info.get("email"):
            _uname = get_or_create_sso_user(_info["email"], _info.get("name", ""))
            st.session_state.logged_in = True
            st.session_state.username  = _uname
            st.session_state.role      = "patient"
            st.query_params.clear()       # remove ?code=&state= first
            _save_session(_uname, "patient")  # then set ?sid= so refresh works
        else:
            st.session_state["_sso_error"] = "Google sign-in failed. Please try again."
            st.query_params.clear()
        st.rerun()

_boot_gate.empty()


# ════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ════════════════════════════════════════════════════════════════════════════

PATIENT_THEME = dict(
    primary="#4F46E5", p_dark="#3730A3", p_light="#EEF2FF",
    secondary="#7C3AED", accent="#A78BFA",
    sidebar_a="#1E1B4B", sidebar_b="#3730A3",
    t1="#1E1B4B", t2="#4B5563", t3="#9CA3AF",
    border="#E0E7FF", bg="#F5F3FF",
    shadow_rgb="79,70,229",
    chart_colors=["#4F46E5","#7C3AED","#A78BFA","#06B6D4","#F59E0B","#EF4444"],
)

ADMIN_THEME = dict(
    primary="#1B3A6B", p_dark="#0F2447", p_light="#EBF2FB",
    secondary="#2B5BA8", accent="#4A90D9",
    sidebar_a="#0F2447", sidebar_b="#1B3A6B",
    t1="#1B2B44", t2="#4A5568", t3="#8492A6",
    border="#D0DBF0", bg="#F0F4FB",
    chart_colors=["#2B5BA8","#1B3A6B","#4A90D9","#7EB8F0","#F59E0B","#DC2626"],
)


# ════════════════════════════════════════════════════════════════════════════
# CHART LAYOUT
# ════════════════════════════════════════════════════════════════════════════

_CHART_CFG = {"displayModeBar": False, "responsive": True}


def chart_layout(t: dict, height: int = 380, title: str = "") -> dict:
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=t["t2"]),
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=t["t3"])),
        yaxis=dict(gridcolor="rgba(0,0,0,.05)", zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(size=11, color=t["t3"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1E293B", font_size=12, font_family="Inter, sans-serif",
                        bordercolor="rgba(0,0,0,0)"),
        bargap=0.35,
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=13, color=t["t1"], family="Plus Jakarta Sans, sans-serif"),
            x=0, xanchor="left",
        )
        layout["margin"] = dict(l=0, r=10, t=36, b=10)
    return layout


def pchart(fig, key: str = None):
    """Render a Plotly chart without the toolbar."""
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG,
                    key=key if key else None)


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════

def inject_global_css(t: dict):
    p   = t["primary"]; pd2 = t["p_dark"]; pl  = t["p_light"]
    s   = t["secondary"]
    sa  = t["sidebar_a"]; sb  = t["sidebar_b"]
    t1  = t["t1"]; t2  = t["t2"]; t3  = t["t3"]
    bdr = t["border"]; bg  = t["bg"]
    shd = t.get("shadow_rgb", "0,108,73")

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}
    h1, h2, h3, .display-font {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}

    /* Force light mode & background */
    html, body {{ background: {bg} !important; color: {t1} !important; }}
    .stApp, [data-testid="stAppViewContainer"] {{
        background: {bg} !important; color: {t1} !important;
    }}
    .main .block-container {{ color: {t1} !important; }}
    .main .block-container p,
    .main .block-container span,
    .main .block-container label {{ color: {t1}; }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {{ color: {t1} !important; }}
    [data-testid="stMarkdownContainer"] p {{ color: {t1} !important; }}

    #MainMenu, footer, header {{ visibility: hidden !important; }}
    .block-container {{ padding: 1.8rem 2rem 3rem !important; max-width: 1440px; }}

    /* ── SIDEBAR ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {sa} 0%, {sb} 100%) !important;
        border-right: none !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.2);
    }}
    [data-testid="stSidebar"] > div p,
    [data-testid="stSidebar"] > div span,
    [data-testid="stSidebar"] > div label,
    [data-testid="stSidebar"] > div div {{ color: rgba(255,255,255,.88) !important; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.12) !important; }}
    [data-testid="stSidebar"] .stButton > button {{
        background: rgba(255,255,255,.08) !important;
        border: 1px solid rgba(255,255,255,.2) !important;
        color: rgba(255,255,255,.8) !important;
        border-radius: 10px !important; width: 100% !important;
        padding: 10px 14px !important; font-weight: 500 !important;
        font-size: .85rem !important; text-align: left !important;
        transition: all .2s !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,.18) !important;
        color: #fff !important; transform: translateX(2px) !important;
    }}

    /* ── TABS (pill style) ───────────────────────────────── */
    [data-baseweb="tab-list"] {{
        background: {pl} !important; border-radius: 12px !important;
        padding: 4px !important; gap: 2px !important;
        border: 1px solid {bdr} !important;
        overflow-x: auto !important; scrollbar-width: none;
    }}
    [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
    [data-baseweb="tab"] {{
        border-radius: 9px !important; padding: 9px 15px !important;
        font-weight: 600 !important; font-size: .85rem !important;
        color: {t2} !important; background: transparent !important;
        border: none !important; transition: all .18s !important;
        font-family: 'Inter', sans-serif !important;
        flex: 0 0 auto !important; white-space: nowrap !important;
    }}
    [aria-selected="true"][data-baseweb="tab"] {{
        background: linear-gradient(135deg,#059669,#10B981) !important;
        color: #fff !important;
        box-shadow: 0 2px 10px rgba(5,150,105,.3) !important;
    }}
    [data-baseweb="tab-panel"] {{ padding-top: 1.6rem !important; }}

    /* ── PRIMARY BUTTON — emerald green globally ────────── */
    .stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"],
    [data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg,#059669,#10B981) !important;
        color: #fff !important;
        border: none !important; border-radius: 10px !important;
        padding: 12px 24px !important; font-weight: 700 !important;
        font-size: .9rem !important; letter-spacing: .01em !important;
        box-shadow: 0 3px 14px rgba(5,150,105,.28) !important;
        transition: all .18s !important;
        font-family: 'Inter', sans-serif !important;
    }}
    .stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg,#047857,#059669) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(5,150,105,.35) !important;
    }}
    /* ── SECONDARY BUTTON — red outline globally ─────── */
    .stButton > button[kind="secondary"],
    [data-testid="stBaseButton-secondary"],
    [data-testid="baseButton-secondary"] {{
        background: transparent !important; color: #DC2626 !important;
        border: 1.5px solid #FCA5A5 !important; border-radius: 10px !important;
        padding: 10px 20px !important; font-weight: 600 !important;
        font-size: .875rem !important; transition: all .15s !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover {{
        background: #FEF2F2 !important; border-color: #DC2626 !important;
    }}

    /* ── FORM INPUTS ─────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        border-radius: 9px !important; border: 1.5px solid {bdr} !important;
        padding: 10px 14px !important; font-size: .875rem !important;
        background: #FAFCFB !important; color: {t1} !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color .15s, box-shadow .15s !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {p} !important; box-shadow: 0 0 0 3px rgba({shd},.1) !important;
    }}
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        border-radius: 9px !important; border: 1.5px solid {bdr} !important;
        background: #FAFCFB !important;
    }}
    [data-testid="stSlider"] [data-testid="stSliderThumb"] {{
        background: {p} !important; border: 2.5px solid #fff !important;
        box-shadow: 0 0 0 2px rgba(0,108,73,.25) !important;
    }}
    [data-testid="stSlider"] [data-testid="stSliderThumb"]:hover {{
        background: {pd2} !important;
    }}
    /* Toggle */
    [data-testid="stToggle"] svg {{ color: {p} !important; }}

    /* ── ALERTS ──────────────────────────────────────────── */
    [data-testid="stAlert"] {{ border-radius: 10px !important; border: none !important; }}

    /* ── DATA TABLE ──────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important; overflow: hidden !important;
        border: 1px solid {bdr} !important;
    }}

    /* ── SCROLLBAR ───────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {bdr}; border-radius: 3px; }}

    /* ── CUSTOM COMPONENT CLASSES ────────────────────────── */
    .kpi-card {{
        background: #FFFFFF; border-radius: 16px; padding: 20px;
        border: 1px solid {bdr}; box-shadow: 0 1px 4px rgba(0,0,0,.04);
        transition: box-shadow .18s, transform .18s; height: 100%;
    }}
    .kpi-card:hover {{ box-shadow: 0 6px 24px rgba({shd},.12); transform: translateY(-2px); }}

    .patient-overview {{ margin: 20px 0 22px; }}
    .patient-overview-head {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:12px; }}
    .patient-overview-kicker {{ color:{p}; font-size:.67rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:4px; }}
    .patient-overview-title {{ color:{t1}; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.05rem; font-weight:800; letter-spacing:-.02em; }}
    .patient-overview-note {{ color:{t3}; font-size:.72rem; }}
    .patient-overview-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .patient-stat {{ min-width:0; background:#fff; border:1px solid {bdr}; border-radius:14px; padding:16px; box-shadow:0 1px 3px rgba(30,27,75,.035); transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }}
    .patient-stat:hover {{ transform:translateY(-2px); border-color:{t['accent']}; box-shadow:0 9px 24px rgba({shd},.09); }}
    .patient-stat-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }}
    .patient-stat-icon {{ width:34px; height:34px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:{pl}; font-size:17px; }}
    .patient-stat-value {{ color:{t1}; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.28rem; line-height:1.1; font-weight:800; letter-spacing:-.025em; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .patient-stat-label {{ color:{t2}; font-size:.76rem; font-weight:650; margin-top:4px; }}
    .patient-stat-sub {{ color:{t3}; font-size:.68rem; margin-top:7px; line-height:1.4; }}
    .dashboard-nav-label {{ display:flex; align-items:center; gap:8px; margin:0 0 9px; color:{t2}; font-size:.76rem; font-weight:700; }}
    .dashboard-nav-label::after {{ content:''; height:1px; flex:1; background:{bdr}; }}
    @media (max-width: 900px) {{
        .patient-overview-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
        .block-container {{ padding:1.2rem 1rem 2rem !important; }}
    }}
    @media (max-width: 520px) {{
        .patient-overview-head {{ align-items:flex-start; flex-direction:column; gap:3px; }}
        .patient-overview-grid {{ gap:9px; }}
        .patient-stat {{ padding:13px; }}
        .patient-stat-value {{ font-size:1.05rem; }}
        [data-baseweb="tab"] {{ padding:8px 12px !important; font-size:.8rem !important; }}
    }}

    .welcome-banner {{
        background: linear-gradient(130deg, {sa} 0%, {p} 100%);
        border-radius: 20px; padding: 26px 32px; margin-bottom: 24px;
        position: relative; overflow: hidden;
    }}
    .welcome-banner::after {{
        content: ''; position: absolute; right: -40px; bottom: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: rgba(255,255,255,.05);
    }}

    .section-card {{
        background: #FFFFFF; border-radius: 16px;
        border: 1px solid {bdr}; box-shadow: 0 1px 4px rgba(0,0,0,.04);
        padding: 22px;
    }}

    .schip {{
        display: flex; align-items: center; gap: 8px; margin: 20px 0 12px;
    }}
    .schip:first-child {{ margin-top: 0; }}
    .schip-label {{
        font-size: .68rem; font-weight: 700; color: {p};
        text-transform: uppercase; letter-spacing: .1em; white-space: nowrap;
    }}
    .schip-line {{ flex: 1; height: 1px; background: {bdr}; }}

    .login-hero {{
        background: linear-gradient(155deg, {sa} 0%, {pd2} 40%, {p} 100%);
        border-radius: 20px; padding: 44px 40px; min-height: 540px;
        position: relative; overflow: hidden;
        display: flex; flex-direction: column; justify-content: space-between;
    }}
    .login-hero::before {{
        content: ''; position: absolute; top: -80px; right: -80px;
        width: 380px; height: 380px; border-radius: 50%;
        background: rgba(255,255,255,.05);
    }}
    .login-hero::after {{
        content: ''; position: absolute; bottom: -100px; left: -60px;
        width: 260px; height: 260px; border-radius: 50%;
        background: rgba(255,255,255,.04);
    }}
    .login-card {{
        background: #FFFFFF; border-radius: 20px; padding: 36px 32px;
        box-shadow: 0 10px 48px rgba(0,0,0,.12); border: 1px solid {bdr};
    }}
    hr {{ border-color: {bdr} !important; margin: 16px 0 !important; }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def kpi_card(icon_bg: str, icon: str, value: str, label: str, sub: str, t: dict, **_) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div style="width:42px;height:42px;border-radius:12px;display:flex;align-items:center;'
        f'justify-content:center;background:{icon_bg};margin-bottom:14px;font-size:20px;">{icon}</div>'
        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.6rem;font-weight:800;'
        f'color:{t["t1"]};letter-spacing:-.02em;line-height:1.1;margin-bottom:3px;">{value}</div>'
        f'<div style="font-size:.78rem;font-weight:600;color:{t["t2"]};margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:.72rem;color:{t["t3"]};">{sub}</div>'
        f'</div>'
    )


def section_hdr(icon_bg: str, icon: str, title: str, subtitle: str, t: dict) -> str:
    return f"""
<div style="display:flex;align-items:center;gap:12px;margin:0 0 16px;">
  <div style="width:36px;height:36px;border-radius:10px;display:flex;align-items:center;
    justify-content:center;background:{icon_bg};font-size:18px;flex-shrink:0;">{icon}</div>
  <div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.95rem;font-weight:700;
      color:{t['t1']};line-height:1.2;">{title}</div>
    <div style="font-size:.72rem;color:{t['t3']};margin-top:1px;">{subtitle}</div>
  </div>
</div>"""


def welcome_banner(first_name: str, subtitle: str, t: dict,
                   patient_id: str = "", joined: str = "", **_) -> str:
    import datetime as _dt
    _hour = _dt.datetime.now().hour
    _greeting = "Good morning" if _hour < 12 else "Good afternoon" if _hour < 17 else "Good evening"

    pid_chip = (
        f'<span style="background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.2);'
        f'border-radius:9999px;padding:3px 12px;font-size:.72rem;font-weight:700;'
        f'color:#fff;letter-spacing:.03em;">&#127973; {patient_id}</span>'
    ) if patient_id else ""

    joined_chip = (
        f'<span style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);'
        f'border-radius:9999px;padding:3px 12px;font-size:.72rem;color:rgba(255,255,255,.8);">'
        f'&#128197; Member since {joined}</span>'
    ) if joined else ""

    return (
        f'<div style="background:linear-gradient(130deg,#1E1B4B 0%,#3730A3 55%,#4F46E5 100%);'
        f'border-radius:20px;padding:28px 32px;margin-bottom:6px;'
        f'position:relative;overflow:hidden;">'
        f'<div style="position:absolute;right:-70px;top:-70px;width:240px;height:240px;'
        f'border-radius:50%;background:rgba(255,255,255,.04);"></div>'
        f'<div style="position:absolute;right:80px;bottom:-90px;width:180px;height:180px;'
        f'border-radius:50%;background:rgba(255,255,255,.03);"></div>'
        f'<div style="position:relative;z-index:1;">'
        f'<div style="font-size:.7rem;color:rgba(255,255,255,.5);letter-spacing:.1em;'
        f'text-transform:uppercase;margin-bottom:6px;">{_greeting}</div>'
        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.9rem;font-weight:800;'
        f'color:#fff;letter-spacing:-.03em;margin-bottom:6px;">{first_name} &#128075;</div>'
        f'<div style="font-size:.84rem;color:rgba(255,255,255,.65);line-height:1.6;margin-bottom:16px;">'
        f'{subtitle}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{pid_chip}{joined_chip}</div>'
        f'</div>'
        f'</div>'
    )


def status_pill(alert: dict, note: str = "") -> str:
    c = alert["color"]; bg = alert["bg"]
    status_icons = {"SURGE": "🔴", "WARNING": "🟡", "NORMAL": "🟢",
                    "BREACH": "🔴", "OK": "🟢", "EXTENDED": "🟠",
                    "ABOVE BENCHMARK": "🟡", "WITHIN BENCHMARK": "🟢"}
    icon = status_icons.get(alert["status"], "⚪")
    return f"""
<div style="background:{bg};border:1px solid {c}38;border-radius:12px;
  padding:14px 18px;display:flex;align-items:flex-start;gap:12px;margin:8px 0;">
  <span style="font-size:1.2rem;flex-shrink:0;margin-top:1px;">{icon}</span>
  <div>
    <div style="font-weight:700;font-size:.88rem;color:{c};margin-bottom:2px;">{alert['status']}</div>
    <div style="font-size:.82rem;color:{c};opacity:.9;">{alert['message']}{note}</div>
  </div>
</div>"""


def result_hero(los_low, los_high, cost_low, cost_high, severity, dept, t: dict) -> str:
    p = t["primary"]; pd2 = t["p_dark"]
    nights_low = max(0, int(los_low))
    nights_high = max(0, int(los_high))
    night_word = "night" if nights_high == 1 else "nights"
    return f"""
<div style="background:linear-gradient(135deg,{pd2} 0%,{p} 60%,{t['secondary']} 100%);
  border-radius:20px;padding:24px;color:#fff;margin-bottom:14px;">
  <div style="font-size:.72rem;font-weight:700;color:rgba(255,255,255,.55);
    text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Your estimate</div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.95rem;font-weight:600;
    color:rgba(255,255,255,.85);margin-bottom:18px;">{dept} &middot; {severity} condition</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div style="background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.15);
      border-radius:14px;padding:16px;">
      <div style="font-size:.78rem;color:rgba(255,255,255,.65);margin-bottom:8px;">🛏️ How long you'll stay</div>
      <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.6rem;font-weight:800;
        color:#fff;line-height:1;">{nights_low}–{nights_high}</div>
      <div style="font-size:.82rem;color:rgba(255,255,255,.65);margin-top:2px;">{night_word} in hospital</div>
    </div>
    <div style="background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.15);
      border-radius:14px;padding:16px;">
      <div style="font-size:.78rem;color:rgba(255,255,255,.65);margin-bottom:8px;">💰 Estimated cost</div>
      <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.3rem;font-weight:800;
        color:#fff;line-height:1;">Rs. {cost_low:,.0f}</div>
      <div style="font-size:.82rem;color:rgba(255,255,255,.65);margin-top:2px;">to Rs. {cost_high:,.0f}</div>
    </div>
  </div>
</div>"""


def trust_note(t: dict) -> str:
    return (
        f'<div style="background:#FFFBEB;border:1px solid rgba(217,119,6,.25);'
        f'border-left:4px solid #D97706;border-radius:10px;'
        f'padding:13px 16px;display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">'
        f'<span style="font-size:1.1rem;flex-shrink:0;margin-top:1px;">💡</span>'
        f'<div style="font-size:.8rem;color:#92400E;line-height:1.6;">'
        f'These figures are a <strong>rough guide</strong> based on thousands of real hospital visits '
        f'in the Bagmati region. They help you plan and budget — they are <strong>not a guarantee, '
        f'a diagnosis, or a final bill</strong>. Your doctor will give you the exact picture '
        f'once they assess you in person.'
        f'</div></div>'
    )


def password_rules_card(password: str = "") -> None:
    """Render a live password-strength checklist driven by validate_password."""
    _ALL_RULES = [
        "At least 8 characters",
        "At least one uppercase letter (A-Z)",
        "At least one lowercase letter (a-z)",
        "At least one number (0-9)",
        "At least one special character  (!@#$%^&*)",
    ]
    _, errs  = validate_password(password)
    failed   = set(errs)
    rules    = [(r, r not in failed) for r in _ALL_RULES]
    met      = sum(1 for _, ok in rules if ok)
    score    = met / len(rules)
    bar_color = "#DC2626" if score < 0.4 else "#D97706" if score < 0.8 else "#0E6B62"
    bar_label = "Weak" if score < 0.4 else "Fair" if score < 0.8 else "Strong"
    items_html = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
        f'<span aria-hidden="true" style="font-size:16px;color:'
        f'{"#0E6B62" if ok else "#B0BAB7"};">{"✓" if ok else "○"}</span>'
        f'<span style="font-size:.78rem;color:{"#12262A" if ok else "#65746F"};">{label}</span>'
        f'</div>'
        for label, ok in rules
    )
    bar_pct = int(score * 100)
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
    .material-symbols-outlined {{ font-variation-settings:'FILL' 1,'wght' 500,'GRAD' 0,'opsz' 20; vertical-align:middle; }}
    </style>
    <div style="background:#F6F8F7;border:1px solid #E3E7E5;border-radius:12px;
      padding:14px 16px;margin:8px 0 12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;">
        <span style="font-size:.7rem;font-weight:700;color:#65746F;
          text-transform:uppercase;letter-spacing:.07em;">Password strength</span>
        <span style="font-size:.72rem;font-weight:700;color:{bar_color};">{bar_label}</span>
      </div>
      <div style="height:6px;background:#E3E7E5;border-radius:9999px;margin-bottom:11px;overflow:hidden;">
        <div style="height:100%;width:{bar_pct}%;background:{bar_color};
          border-radius:9999px;transition:width .3s ease;"></div>
      </div>
      {items_html}
    </div>
    """, unsafe_allow_html=True)


def disclaimer_box(msg: str) -> str:
    return (
        f'<div style="background:#FFFBEB;border:1px solid rgba(217,119,6,.2);'
        f'border-left:3px solid #D97706;border-radius:10px;padding:12px 16px;'
        f'font-size:.8rem;color:#92400E;line-height:1.55;margin:10px 0;">'
        f'{msg}</div>'
    )


def insight_box(title: str, body: str, t: dict) -> str:
    shd = t.get("shadow_rgb","0,108,73")
    return f"""
<div style="background:{t['p_light']};border:1px solid rgba({shd},.15);border-radius:14px;
  padding:18px 20px;display:flex;align-items:flex-start;gap:14px;margin-top:14px;">
  <div style="width:40px;height:40px;border-radius:10px;background:{t['primary']};
    display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">💡</div>
  <div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;color:{t['t1']};
      margin-bottom:5px;">{title}</div>
    <div style="font-size:.85rem;color:{t['t2']};line-height:1.65;">{body}</div>
  </div>
</div>"""


def season_tip_card(emoji: str, season: str, badge: str, badge_bg: str, badge_color: str,
                    body: str, border_color: str) -> str:
    return f"""
<div style="background:#fff;border:1px solid #D8E8E2;border-left:4px solid {border_color};
  border-radius:12px;padding:14px 16px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span style="font-size:1.3rem;">{emoji}</span>
    <div>
      <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.88rem;font-weight:700;
        color:#1A2E25;">{season}</div>
      <span style="background:{badge_bg};color:{badge_color};font-size:.65rem;font-weight:700;
        padding:2px 8px;border-radius:9999px;">{badge}</span>
    </div>
  </div>
  <p style="font-size:.77rem;color:#4A6358;line-height:1.55;margin:0">{body}</p>
</div>"""


# ════════════════════════════════════════════════════════════════════════════
# SESSION BAR
# ════════════════════════════════════════════════════════════════════════════

def render_session_bar(notifs=None, n_total=0):
    """Top bar: session status | [bell] | Sign Out."""
    expires   = st.session_state.get("_session_expires", 0.0)
    remaining = max(0, int(expires - _time.time()))
    mins      = remaining // 60

    if mins <= 5:
        bg = "#FEF2F2"; border = "#FCA5A5"; color = "#991B1B"
        icon = "&#128308;"
        msg  = f"Session expires in <strong>{mins} min</strong> &mdash; any action resets the timer"
    elif mins <= 15:
        bg = "#FFFBEB"; border = "#FCD34D"; color = "#92400E"
        icon = "&#9888;&#65039;"
        msg  = f"Session expires in <strong>{mins} min</strong>"
    else:
        bg = "#F0FDF4"; border = "#BBF7D0"; color = "#166534"
        icon = "&#128274;"
        msg  = "Session active &mdash; auto-logout after <strong>1 hour</strong> of inactivity"

    if notifs is not None:
        bar_col, bell_col, logout_col = st.columns([6, 1, 1])
    else:
        bar_col, logout_col = st.columns([6, 1])
        bell_col = None

    with bar_col:
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
            f'padding:9px 16px;font-size:.82rem;color:{color};'
            f'display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="font-size:.95rem;">{icon}</span><span>{msg}</span></div>',
            unsafe_allow_html=True,
        )

    if bell_col is not None:
        with bell_col:
            badge = f" ({n_total})" if n_total else ""
            n_urgent = sum(1 for n in (notifs or []) if n["severity"] in ("urgent", "warning"))
            with st.popover(f"🔔{badge}", use_container_width=True):
                st.markdown(
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.92rem;'
                    f'font-weight:800;color:#1E1B4B;margin-bottom:12px;">Notifications'
                    f'{"&nbsp;&nbsp;<span style=\'background:#EF4444;color:#fff;font-size:.65rem;font-weight:700;padding:2px 7px;border-radius:9999px;\'>" + str(n_urgent) + " urgent</span>" if n_urgent else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if not notifs:
                    st.markdown(
                        '<div style="text-align:center;padding:20px 0;font-size:.82rem;color:#9CA3AF;">'
                        '✅ All clear — nothing to flag right now.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for n in notifs:
                        st.markdown(_notif_card_html(n), unsafe_allow_html=True)

    with logout_col:
        if st.button("Sign Out", key="top_signout", use_container_width=True):
            _clear_session()
            for k in ["logged_in", "username", "role", "page", "profile_msg"]:
                st.session_state[k] = (
                    False if k == "logged_in" else
                    "dashboard" if k == "page" else ""
                )
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(t: dict):
    uname      = st.session_state.username
    role       = st.session_state.role
    initials   = (uname[:2].upper() if len(uname) >= 2 else (uname.upper() + "?"))
    role_label = "Patient Portal" if role == "patient" else "Admin Console"
    cur_page   = st.session_state.get("page", "dashboard")

    # Load avatar for sidebar (48 px thumbnail — ~3 KB base64)
    if role == "patient":
        b64 = get_avatar_thumb_b64(uname, 48)
        if b64:
            av_html = (f'<img src="data:image/jpeg;base64,{b64}" '
                       f'style="width:42px;height:42px;border-radius:50%;'
                       f'object-fit:cover;border:2px solid rgba(255,255,255,.3);">')
        else:
            av_html = (f'<div style="width:42px;height:42px;border-radius:50%;'
                       f'background:linear-gradient(135deg,{t["secondary"]},{t["primary"]});'
                       f'display:flex;align-items:center;justify-content:center;'
                       f'font-weight:700;font-size:.82rem;color:#fff;flex-shrink:0;">{initials}</div>')
    else:
        av_html = (f'<div style="width:42px;height:42px;border-radius:50%;'
                   f'background:linear-gradient(135deg,{t["secondary"]},{t["primary"]});'
                   f'display:flex;align-items:center;justify-content:center;'
                   f'font-weight:700;font-size:.82rem;color:#fff;flex-shrink:0;">{initials}</div>')

    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div style="padding:22px 16px 16px;border-bottom:1px solid rgba(255,255,255,.1);">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.15);
              border:1px solid rgba(255,255,255,.25);display:flex;align-items:center;
              justify-content:center;font-size:18px;">&#10133;</div>
            <span style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
              font-size:.95rem;color:#fff;">Hospital Intelligence System</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # User card
        st.markdown(f"""
        <div style="padding:16px;border-bottom:1px solid rgba(255,255,255,.1);">
          <div style="display:flex;align-items:center;gap:12px;">
            {av_html}
            <div>
              <div style="font-weight:600;font-size:.88rem;color:#fff;line-height:1.2;">{uname}</div>
              <div style="display:inline-block;margin-top:3px;padding:2px 8px;border-radius:9999px;
                background:rgba(255,255,255,.12);font-size:.62rem;font-weight:600;
                color:rgba(255,255,255,.75);letter-spacing:.05em;text-transform:uppercase;">
                {role_label}
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Nav label
        st.markdown("""
        <div style="padding:14px 16px 6px;font-size:.62rem;font-weight:700;
          color:rgba(255,255,255,.35);text-transform:uppercase;letter-spacing:.12em;">
          Navigation
        </div>""", unsafe_allow_html=True)

        # Nav items
        def nav_btn(label, page_key, icon):
            active = cur_page == page_key
            bg = "rgba(255,255,255,.22)" if active else "rgba(255,255,255,.06)"
            border = "rgba(255,255,255,.35)" if active else "rgba(255,255,255,.12)"
            st.markdown(f"""
            <style>
            div[data-testid="stButton"] > button[title="{page_key}"] {{
              background:{bg} !important; border:1px solid {border} !important;
              color:#fff !important; text-align:left !important;
            }}
            </style>""", unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", use_container_width=True,
                         key=f"nav_{page_key}", help=page_key):
                st.session_state.page = page_key
                st.rerun()

        nav_btn("Dashboard", "dashboard", "🏠")
        if role == "patient":
            nav_btn("My Profile", "profile", "👤")

        st.markdown("""
        <div style="padding:12px 16px 6px;margin-top:6px;font-size:.62rem;font-weight:700;
          color:rgba(255,255,255,.25);text-transform:uppercase;letter-spacing:.12em;
          border-top:1px solid rgba(255,255,255,.08);">
          Account
        </div>""", unsafe_allow_html=True)

        if st.button("🚪  Sign Out", use_container_width=True, key="nav_logout"):
            _clear_session()
            for k in ["logged_in", "username", "role", "page", "profile_msg"]:
                st.session_state[k] = (
                    False if k == "logged_in" else
                    "dashboard" if k == "page" else ""
                )
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(255,255,255,.05);border-radius:12px;padding:14px;
          font-size:.73rem;color:rgba(255,255,255,.4);line-height:1.65;margin:0 4px;">
          <div style="font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
            color:rgba(255,255,255,.25);margin-bottom:6px;">About</div>
          Hospital Intelligence System<br>
          Bagmati Region, Nepal<br>
          Synthetic Data &#183; 2021&#8211;2024<br>
          Coventry University
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════════════════════════════════

def page_login():
    t = PATIENT_THEME
    inject_global_css(t)

    view = st.session_state.login_view  # "signin" | "register"

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
    .material-symbols-outlined { font-family:'Material Symbols Outlined'; font-variation-settings:'FILL' 0,'wght' 450,'GRAD' 0,'opsz' 22; vertical-align:middle; }

    [data-testid="stSidebar"] { display:none !important; }
    [data-testid="stAppViewContainer"] { background:#FBF9F4 !important; }
    #MainMenu, footer { visibility:hidden !important; }
    header[data-testid="stHeader"] { display:none !important; }
    .block-container { padding:0 !important; max-width:none !important; }
    .block-container > div[data-testid="stVerticalBlock"] { gap:0 !important; }

    .mobile-brand-bar { display:none; align-items:center; gap:10px; padding:22px 22px 0; }
    div[data-testid="stElementContainer"]:has(.mobile-brand-bar) { display:none; }
    .mobile-brand-bar .brand-mark {
      width:34px; height:34px; border-radius:10px; color:#fff; flex-shrink:0;
      background:linear-gradient(135deg,#0E6B62,#17A398);
      display:flex; align-items:center; justify-content:center;
    }
    .mobile-brand-bar .brand-word {
      font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1rem; color:#12262A;
    }

    /* ── SHELL ─────────────────────────────────────────────────────────── */
    .st-key-auth_shell {
      --teal-900:#0B4A44; --teal-700:#0E6B62; --teal-600:#12897B; --teal-500:#17A398;
      --teal-100:#E3F4F1; --teal-50:#F1FAF8;
      --indigo-500:#5B67CE; --indigo-100:#EEF0FC; --coral-500:#E1705A;
      --ink-900:#12262A; --ink-700:#3E4C48; --ink-500:#65746F; --ink-300:#A9B6B2;
      --canvas:#FBF9F4; --surface:#FFFFFF; --border:#E3E7E5; --border-strong:#CBD5D2;
      --danger:#DC2626; --success:#0E9F6E; --warning:#D97706;
      --radius-sm:10px; --radius-md:14px; --radius-lg:20px; --radius-xl:24px;
      --shadow-md:0 10px 30px rgba(16,42,38,.08);
      --shadow-lg:0 30px 70px rgba(16,42,38,.16);
      width:100%; min-height:100dvh; background:var(--canvas);
    }
    .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
      min-height:100dvh; align-items:stretch; gap:0 !important;
    }
    .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
      min-width:0; display:flex;
    }
    .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
      width:100%; display:flex; flex-direction:column;
    }

    /* ── HERO PANEL ────────────────────────────────────────────────────── */
    .st-key-hero_panel {
      flex:1; position:relative; overflow:hidden; min-height:100dvh;
      background:
        radial-gradient(circle at 82% 12%, rgba(255,255,255,.07), transparent 40%),
        radial-gradient(circle at 8% 92%, rgba(0,0,0,.14), transparent 45%),
        linear-gradient(160deg,var(--teal-900) 0%,var(--teal-700) 55%,var(--teal-500) 100%);
      padding:clamp(2.75rem,4vw,4.5rem);
    }
    .hero-inner { height:100%; max-width:590px; margin:0 auto; display:flex; flex-direction:column; justify-content:space-between; position:relative; z-index:1; }
    .hero-brand-row { display:flex; align-items:center; gap:12px; }
    .hero-brand-mark {
      width:48px; height:48px; border-radius:14px; color:#fff; flex-shrink:0;
      background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
      display:flex; align-items:center; justify-content:center;
    }
    .hero-brand-word { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.28rem; color:#fff; }

    .hero-eyebrow {
      display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:9999px;
      background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18);
      color:#D7F5EE; font-size:.65rem; font-weight:700; letter-spacing:.08em; margin-bottom:20px;
    }
    .hero-eyebrow .dot {
      width:6px; height:6px; border-radius:50%; background:#6EE7C6;
      box-shadow:0 0 0 4px rgba(110,231,198,.18);
    }
    .hero-headline {
      font-family:'Plus Jakarta Sans',sans-serif; font-weight:700;
      font-size:clamp(2.5rem,3.15vw,3.45rem); line-height:1.08; letter-spacing:-.04em;
      color:#F4FBF9; margin-bottom:18px; max-width:570px;
    }
    .hero-copy { font-size:1.02rem; color:rgba(244,251,249,.78); line-height:1.65; max-width:520px; margin-bottom:30px; }

    .hero-visual-card {
      background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.16);
      border-radius:var(--radius-lg); padding:20px 24px; max-width:500px;
      box-shadow:0 18px 45px rgba(5,55,50,.13);
    }
    .hvc-head { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
    .hvc-head .material-symbols-outlined { color:#8FE3CE; font-size:20px; }
    .hvc-head > span[aria-hidden] { color:#6EE7C6; font-size:.82rem; }
    .hvc-head-label { font-size:.78rem; font-weight:700; color:#EAFBF6; flex:1; }
    .hvc-live-dot { width:7px; height:7px; border-radius:50%; background:#6EE7C6; box-shadow:0 0 0 3px rgba(110,231,198,.2); }
    .hvc-row { display:flex; align-items:center; gap:12px; min-height:44px; padding:8px 0; }
    .hvc-row + .hvc-row { border-top:1px solid rgba(255,255,255,.1); }
    .hvc-row .material-symbols-outlined { color:#BFEFE1; font-size:18px; flex-shrink:0; }
    .hvc-row > span[aria-hidden] { width:12px; color:#6EE7C6; font-size:.75rem; text-align:center; flex-shrink:0; }
    .hvc-row-label { font-size:.8rem; color:rgba(244,251,249,.72); flex:1; }
    .hvc-row-val { font-family:'Plus Jakarta Sans',sans-serif; min-width:46px; text-align:right; font-size:.88rem; font-weight:800; color:#fff; }
    .hvc-bar-track { width:64px; height:5px; border-radius:9999px; background:rgba(255,255,255,.16); overflow:hidden; flex-shrink:0; }
    .hvc-bar-fill { height:100%; border-radius:9999px; background:#6EE7C6; }
    @media (prefers-reduced-motion:no-preference) {
      .hero-visual-card { animation:hvcFloat 6s ease-in-out infinite; }
    }
    @keyframes hvcFloat { 0%,100%{ transform:translateY(0);} 50%{ transform:translateY(-7px);} }

    .hero-footnote { font-size:.7rem; color:rgba(244,251,249,.55); letter-spacing:.02em; margin-top:22px; }

    /* ── AUTH REGION / CARD ────────────────────────────────────────────── */
    .st-key-auth_region { flex:1; min-height:100dvh; display:flex; align-items:center; justify-content:center; padding:clamp(28px,4vw,64px); }
    .st-key-auth_region > div[data-testid="stVerticalBlock"] { width:100%; align-items:center; justify-content:center; }
    .st-key-auth_card {
      box-sizing:border-box; width:min(100%, 500px); background:var(--surface); border:1px solid var(--border);
      border-radius:var(--radius-xl); box-shadow:0 24px 60px rgba(16,42,38,.12); padding:36px 40px;
    }
    .st-key-auth_card > div[data-testid="stVerticalBlock"] { gap:.55rem !important; }
    .st-key-auth_card [data-testid="stForm"] [data-testid="stVerticalBlock"] { gap:.65rem !important; }
    .st-key-auth_card:has(.register-step) { width:min(100%, 520px); }

    .auth-eyebrow {
      display:flex; align-items:center; justify-content:center; gap:6px;
      font-size:.68rem; font-weight:800; letter-spacing:.09em; color:var(--teal-700);
      margin-bottom:4px; text-transform:uppercase;
    }
    .auth-title {
      font-family:'Plus Jakarta Sans',sans-serif; font-size:1.85rem; font-weight:800;
      color:var(--ink-900); text-align:center; letter-spacing:-.02em; line-height:1.2;
    }
    .auth-subtitle { font-size:.88rem; color:var(--ink-500); text-align:center; margin-top:4px; margin-bottom:14px; }

    /* segmented control */
    .st-key-seg_track { background:var(--teal-50); border:1px solid var(--border); border-radius:14px; padding:4px; margin-bottom:10px; }
    .st-key-seg_track [data-testid="stHorizontalBlock"] { gap:4px !important; }
    .st-key-seg_track [data-testid="stColumn"] { min-width:0 !important; }
    .st-key-seg_track .stButton>button {
      border-radius:10px !important; min-height:44px !important; font-size:.84rem !important;
      font-weight:700 !important; transition:all .15s ease !important;
    }
    .st-key-seg_track .stButton>button[kind="primary"] {
      background:var(--teal-700) !important; color:#fff !important; border:0 !important;
      box-shadow:0 4px 14px rgba(14,107,98,.25) !important;
    }
    .st-key-seg_track .stButton>button[kind="secondary"] {
      background:transparent !important; border:0 !important; color:var(--ink-500) !important; box-shadow:none !important;
    }
    .st-key-seg_track .stButton>button[kind="secondary"]:hover {
      background:rgba(14,107,98,.08) !important; color:var(--teal-700) !important;
    }

    /* labels + inputs */
    .st-key-auth_card [data-testid="stWidgetLabel"] p {
      color:var(--ink-700) !important; font-weight:600 !important; font-size:.79rem !important;
    }
    .st-key-auth_card .stTextInput input {
      min-height:48px !important; border-radius:var(--radius-sm) !important;
      border:1.5px solid var(--border-strong) !important; background:var(--canvas) !important;
      font-size:.92rem !important; color:var(--ink-900) !important;
      transition:border-color .15s ease, box-shadow .15s ease, background-color .15s ease !important;
    }
    .st-key-auth_card .stTextInput input:hover:not(:focus) { border-color:var(--ink-300) !important; }
    .st-key-auth_card .stTextInput input:focus {
      border-color:var(--teal-600) !important; background:#fff !important;
      box-shadow:0 0 0 4px rgba(14,107,98,.12) !important; outline:none !important;
    }
    .st-key-auth_card .stTextInput input:disabled { opacity:.55 !important; }
    .st-key-auth_card [data-testid="stTextInputRootElement"] svg,
    .st-key-auth_card [data-testid="stTextInputRootElement"] .material-symbols-outlined { color:var(--ink-300) !important; }
    .st-key-auth_card [data-testid="stTextInputRootElement"]:focus-within svg,
    .st-key-auth_card [data-testid="stTextInputRootElement"]:focus-within .material-symbols-outlined { color:var(--teal-600) !important; }

    .st-key-auth_card [data-testid="stCheckbox"] label p { font-size:.8rem !important; color:var(--ink-700) !important; }

    /* buttons */
    .st-key-auth_card .stButton>button[kind="primary"],
    .st-key-auth_card [data-testid="stFormSubmitButton"] button {
      background:linear-gradient(135deg,var(--teal-700),var(--teal-600)) !important;
      border:0 !important; color:#fff !important; min-height:48px !important;
      border-radius:var(--radius-sm) !important; font-weight:700 !important; font-size:.92rem !important;
      box-shadow:0 10px 26px rgba(14,107,98,.26) !important;
      transition:transform .15s ease, box-shadow .15s ease !important; width:100% !important;
    }
    .st-key-auth_card .stButton>button[kind="primary"]:hover,
    .st-key-auth_card [data-testid="stFormSubmitButton"] button:hover {
      transform:translateY(-1px) !important; box-shadow:0 14px 34px rgba(14,107,98,.32) !important;
    }
    .st-key-auth_card .stButton>button:disabled,
    .st-key-auth_card [data-testid="stFormSubmitButton"] button:disabled {
      opacity:.55 !important; transform:none !important; box-shadow:none !important; cursor:not-allowed !important;
    }
    .st-key-auth_card [data-testid="stForm"] { border:0 !important; padding:0 !important; }

    .google-btn {
      display:flex; align-items:center; justify-content:center; gap:10px;
      border:1.5px solid var(--border-strong); border-radius:var(--radius-sm);
      padding:12px 16px; min-height:48px; background:#fff; font-size:.88rem; font-weight:600;
      color:var(--ink-700); text-decoration:none; transition:box-shadow .15s ease, border-color .15s ease;
    }
    .google-btn:hover { border-color:var(--ink-300); box-shadow:var(--shadow-md); }

    .auth-divider { display:flex; align-items:center; gap:10px; margin:10px 0; }
    .auth-divider .line { flex:1; height:1px; background:var(--border); }
    .auth-divider span { font-size:.72rem; color:var(--ink-300); }

    .auth-link { color:var(--teal-700); font-size:.78rem; font-weight:600; text-decoration:none; }
    .auth-link:hover { text-decoration:underline; }
    .st-key-login_options [data-testid="stHorizontalBlock"] { align-items:center; }
    .st-key-login_options [data-testid="stColumn"] { min-width:0 !important; }
    .st-key-reg_actions [data-testid="stHorizontalBlock"] { align-items:stretch; }
    .st-key-reg_actions [data-testid="stColumn"] { min-width:0 !important; }

    .security-note { display:flex; align-items:center; justify-content:center; gap:6px; font-size:.72rem; color:var(--ink-500); margin-top:8px; }
    .security-note .material-symbols-outlined { font-size:15px; color:var(--ink-300); }

    .success-banner {
      background:#ECFDF5; border:1px solid rgba(14,159,110,.28); border-radius:var(--radius-sm);
      padding:11px 14px; display:flex; align-items:center; gap:10px; font-size:.82rem; color:#065F46; margin-bottom:16px;
    }

    .auth-section-label {
      font-size:.66rem; font-weight:700; color:var(--ink-500); text-transform:uppercase;
      letter-spacing:.09em; margin:18px 0 8px;
    }

    .auth-view-marker { display:none; }
    @media (prefers-reduced-motion:no-preference) {
      .st-key-auth_card:has(.view-signin) {
        animation:authViewFromLeft 420ms cubic-bezier(.2,.85,.25,1) both;
      }
      .st-key-auth_card:has(.view-register) {
        animation:authViewFromRight 420ms cubic-bezier(.2,.85,.25,1) both;
      }
      .st-key-auth_card:has(.step-one) {
        animation:authViewFromRight 420ms cubic-bezier(.2,.85,.25,1) both;
      }
      .st-key-auth_card:has(.step-two) {
        animation:authViewFromRight 420ms cubic-bezier(.2,.85,.25,1) both;
      }
    }
    @media (prefers-reduced-motion:reduce) {
      .st-key-auth_card *, .st-key-seg_track * { animation:none !important; transition:none !important; }
    }
    @keyframes authViewFromLeft {
      from { opacity:0; transform:translateX(-16px) scale(.992); }
      to { opacity:1; transform:translateX(0) scale(1); }
    }
    @keyframes authViewFromRight {
      from { opacity:0; transform:translateX(16px) scale(.992); }
      to { opacity:1; transform:translateX(0) scale(1); }
    }

    .auth-footer {
      text-align:center; font-size:.72rem; color:var(--ink-300); margin-top:22px;
      padding-top:16px; border-top:1px solid var(--border);
      display:flex; align-items:center; justify-content:center; gap:6px;
    }

    button:focus-visible, input:focus-visible, a:focus-visible {
      outline:3px solid rgba(23,163,152,.45) !important; outline-offset:2px !important;
    }

    @media (min-width:768px) and (max-width:1099px) {
      .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { flex-basis:38% !important; }
      .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child { flex-basis:62% !important; }
      .st-key-hero_panel { padding:32px 26px; }
      .hero-headline { font-size:2rem; }
      .hero-copy { font-size:.9rem; }
      .hero-visual-card { padding:16px; }
      .st-key-auth_region { padding:24px; }
    }
    @media (min-width:1100px) and (max-height:820px) {
      .st-key-hero_panel { padding:32px 48px; }
      .hero-headline { font-size:2.45rem; margin-bottom:10px; }
      .hero-copy { font-size:.92rem; line-height:1.5; margin-bottom:18px; }
      .hero-visual-card { padding:15px 20px; }
      .hvc-head { margin-bottom:8px; }
      .hvc-row { padding:6px 0; }
      .hero-footnote { margin-top:12px; }
      .st-key-auth_region { padding:20px 40px; }
      .st-key-auth_card { padding:24px 34px; }
      .auth-subtitle { margin-bottom:12px; }
      .st-key-seg_track { margin-bottom:10px; }
      .auth-divider { margin:8px 0; }
      .auth-footer { margin-top:10px; padding-top:10px; }
    }
    @media (max-width:767px) {
      div[data-testid="stElementContainer"]:has(.mobile-brand-bar) { display:block; }
      .mobile-brand-bar { display:flex !important; }
      .st-key-auth_shell, .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { min-height:auto; }
      .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { display:flex; flex-direction:column-reverse; }
      .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width:100% !important; flex:1 1 100% !important;
      }
      .st-key-hero_panel { display:none !important; }
      .st-key-auth_region { min-height:auto; padding:20px 16px 40px; }
      .st-key-auth_card, .st-key-auth_card:has(.register-step) {
        width:100%; border-radius:var(--radius-lg); padding:26px 28px;
      }
      .auth-title { font-size:1.55rem; }
      .auth-subtitle { margin-bottom:14px; }
      .st-key-seg_track { margin-bottom:12px; }
      .st-key-seg_track [data-testid="stHorizontalBlock"],
      .st-key-login_options [data-testid="stHorizontalBlock"],
      .st-key-reg_actions [data-testid="stHorizontalBlock"] {
        display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
      }
      .st-key-seg_track [data-testid="stColumn"] { width:50% !important; flex:1 1 50% !important; }
      .st-key-login_options [data-testid="stColumn"] { width:50% !important; flex:1 1 50% !important; }
      .st-key-reg_actions [data-testid="stColumn"]:first-child { width:34% !important; flex:1 1 34% !important; }
      .st-key-reg_actions [data-testid="stColumn"]:last-child { width:66% !important; flex:1 1 66% !important; }
      .auth-divider { margin:10px 0; }
      .auth-footer { margin-top:12px; padding-top:12px; }
      .security-note { margin-top:8px; }
    }

    /* Premium Clinical Editorial foundation */
    [data-testid="stAppViewContainer"], .st-key-auth_shell { background:#F7F8F5 !important; }
    .auth-page-header { height:72px; display:flex; align-items:center; border-bottom:1px solid #E3E8E5; background:rgba(247,248,245,.94); }
    .auth-header-inner { width:min(1280px, calc(100% - 64px)); margin:0 auto; display:flex; align-items:center; justify-content:space-between; }
    .auth-brand { display:flex; align-items:center; gap:11px; color:#102A2A; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.05rem; font-weight:800; letter-spacing:-.02em; }
    .auth-brand-mark { width:38px; height:38px; display:block; }
    .auth-header-actions { display:flex; align-items:center; gap:28px; font-size:.78rem; }
    .auth-header-actions a { color:#53615E; text-decoration:none; font-weight:600; }
    .auth-header-actions a:hover { color:#0F766E; }
    .workspace-status { display:flex; align-items:center; gap:8px; color:#63706D; }
    .workspace-status > span { width:7px; height:7px; border-radius:50%; background:#0F766E; box-shadow:0 0 0 3px #E6F0EC; }

    .st-key-auth_shell { min-height:calc(100dvh - 72px); }
    .st-key-auth_shell > [data-testid="stLayoutWrapper"] { width:min(1280px, calc(100% - 64px)); margin:0 auto; }
    .st-key-auth_shell > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { min-height:calc(100dvh - 72px); }
    .st-key-hero_panel, .st-key-auth_region { min-height:calc(100dvh - 72px); }
    .st-key-hero_panel { background:transparent; padding:clamp(34px,4vw,60px) clamp(12px,3vw,42px) clamp(30px,3vw,48px) 0; }
    .hero-inner { max-width:680px; margin:0; justify-content:center; gap:clamp(18px,3vh,30px); }
    .hero-copy-block { max-width:620px; }
    .hero-kicker { margin-bottom:13px; color:#0F766E; font-size:.7rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .hero-headline { max-width:620px; margin:0 0 18px; color:#102A2A; font-size:clamp(2.65rem,3.7vw,4rem); line-height:1.03; letter-spacing:-.05em; }
    .hero-copy { max-width:540px; margin:0; color:#63706D; font-size:1rem; line-height:1.65; }
    .care-visual { width:min(100%,660px); }
    .care-visual svg { display:block; width:100%; height:auto; }
    .hero-trust { display:flex; align-items:center; gap:9px; color:#53615E; font-size:.76rem; font-weight:600; }
    .hero-trust > span { width:22px; height:1px; background:#0F766E; }

    .st-key-auth_region { padding:24px 0 24px clamp(28px,4vw,58px); }
    .st-key-auth_card, .st-key-auth_card:has(.register-step) { width:min(100%,470px); border:1px solid #E1E7E4; border-radius:16px; box-shadow:0 18px 50px rgba(24,52,48,.07); padding:36px 40px; }
    .st-key-auth_card > div[data-testid="stVerticalBlock"] { gap:.48rem !important; }
    .auth-eyebrow { justify-content:flex-start; margin:0 0 3px; color:#0F766E; font-size:.66rem; letter-spacing:.11em; }
    .auth-title { text-align:left; color:#102A2A; font-size:1.75rem; letter-spacing:-.035em; }
    .auth-subtitle { text-align:left; margin:2px 0 15px; color:#63706D; font-size:.84rem; }
    .st-key-seg_track { padding:0; margin:0 0 12px; border:0; border-bottom:1px solid #DDE4E1; border-radius:0; background:transparent; }
    .st-key-seg_track .stButton>button { min-height:40px !important; padding:0 8px !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important; }
    .st-key-auth_card .st-key-seg_track .stButton>button[kind="primary"] { color:#0F766E !important; background:#F1F7F4 !important; border:0 !important; box-shadow:none !important; animation:activeTabWash 460ms cubic-bezier(.2,.8,.2,1) both; }
    .st-key-auth_card .st-key-seg_track .stButton>button[kind="secondary"] { color:#71807C !important; background:transparent !important; border:0 !important; box-shadow:none !important; }
    .st-key-seg_track .stButton>button p { color:inherit !important; }
    .st-key-seg_track { position:relative; overflow:visible; }
    .st-key-seg_track::after {
      content:""; position:absolute; z-index:2; left:0; bottom:-1px; width:50%; height:2px;
      border-radius:999px; background:#0F766E; pointer-events:none;
      box-shadow:0 2px 9px rgba(15,118,110,.32);
    }
    @media (prefers-reduced-motion:no-preference) {
      .st-key-seg_track:has([data-testid="stColumn"]:first-child button[kind="primary"])::after {
        animation:tabIndicatorLeft 480ms cubic-bezier(.2,.9,.25,1) both;
      }
      .st-key-seg_track:has([data-testid="stColumn"]:last-child button[kind="primary"])::after {
        animation:tabIndicatorRight 480ms cubic-bezier(.2,.9,.25,1) both;
      }
    }
    @media (prefers-reduced-motion:reduce) {
      .st-key-seg_track:has([data-testid="stColumn"]:first-child button[kind="primary"])::after { transform:translateX(0); }
      .st-key-seg_track:has([data-testid="stColumn"]:last-child button[kind="primary"])::after { transform:translateX(100%); }
    }
    @keyframes tabIndicatorLeft {
      0% { opacity:.35; transform:translateX(100%) scaleX(.55); }
      62% { opacity:1; transform:translateX(-4%) scaleX(1.08); }
      100% { opacity:1; transform:translateX(0) scaleX(1); }
    }
    @keyframes tabIndicatorRight {
      0% { opacity:.35; transform:translateX(0) scaleX(.55); }
      62% { opacity:1; transform:translateX(104%) scaleX(1.08); }
      100% { opacity:1; transform:translateX(100%) scaleX(1); }
    }
    @keyframes activeTabWash {
      from { background:transparent; }
      to { background:#F1F7F4; }
    }
    .st-key-seg_track .stButton>button[kind="secondary"]:hover { color:#102A2A !important; background:transparent !important; }
    .google-btn { min-height:48px; padding:0 16px; border:1px solid #D7E0DC; border-radius:10px; color:#253C39; font-size:.84rem; }
    .google-btn:hover { border-color:#97ADA6; box-shadow:0 5px 15px rgba(24,52,48,.06); }
    .st-key-auth_card [data-testid="stTextInputRootElement"] { min-height:50px; overflow:hidden; border:1px solid #CAD5D1; border-radius:10px; background:#fff; transition:border-color .18s ease, box-shadow .18s ease; }
    .st-key-auth_card [data-testid="stTextInputRootElement"]:focus-within { border-color:#0F766E; box-shadow:0 0 0 3px rgba(15,118,110,.12); }
    .st-key-auth_card .stTextInput input { min-height:48px !important; border:0 !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important; }
    .st-key-auth_card .stButton>button[kind="primary"], .st-key-auth_card [data-testid="stFormSubmitButton"] button { min-height:50px !important; border-radius:10px !important; background:#0F766E !important; box-shadow:none !important; color:#fff !important; }
    .st-key-auth_card .stButton>button[kind="primary"] p, .st-key-auth_card [data-testid="stFormSubmitButton"] button p { color:inherit !important; }
    .st-key-auth_card .stButton>button[kind="primary"]:hover, .st-key-auth_card [data-testid="stFormSubmitButton"] button:hover { background:#0B655E !important; box-shadow:0 6px 16px rgba(15,118,110,.14) !important; }
    .auth-divider { margin:9px 0; }
    .security-note { justify-content:flex-start; margin-top:8px; color:#71807C; }
    .auth-section-label { margin:12px 0 7px; }

    @media (min-width:1100px) and (max-height:820px) {
      .st-key-hero_panel { padding-top:26px; padding-bottom:26px; }
      .hero-inner { gap:14px; }
      .hero-headline { font-size:3rem; margin-bottom:10px; }
      .hero-copy { line-height:1.5; }
      .care-visual { width:min(92%,540px); }
      .st-key-auth_card, .st-key-auth_card:has(.register-step) { padding:27px 34px; }
    }
    @media (min-width:768px) and (max-width:1099px) {
      .st-key-auth_shell > [data-testid="stLayoutWrapper"] { width:calc(100% - 40px); }
      .st-key-hero_panel { padding-right:24px; }
      .hero-headline { font-size:2.5rem; }
      .care-visual { opacity:.78; }
      .st-key-auth_region { padding-left:24px; }
    }
    @media (max-width:767px) {
      .auth-page-header { height:64px; }
      .auth-header-inner { width:calc(100% - 32px); }
      .auth-header-actions a { display:none; }
      .workspace-status { font-size:0; }
      .workspace-status::after { content:'Secure'; font-size:.72rem; }
      .st-key-auth_shell { min-height:calc(100dvh - 64px); }
      .st-key-auth_shell > [data-testid="stLayoutWrapper"] { width:100%; }
      .st-key-auth_region { min-height:auto; padding:22px 16px 30px; }
      .st-key-auth_card, .st-key-auth_card:has(.register-step) { padding:28px 24px; border-radius:14px; }
      .auth-title { font-size:1.55rem; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-page-header" role="banner">
      <div class="auth-header-inner">
        <div class="auth-brand" aria-label="Hospital Intelligence System">
          <svg class="auth-brand-mark" viewBox="0 0 40 40" role="img" aria-label="Hospital Intelligence System mark">
            <path d="M20 3.5C13 3.5 7.2 7.8 7.2 14.6c0 9.2 12.8 21.9 12.8 21.9s12.8-12.7 12.8-21.9C32.8 7.8 27 3.5 20 3.5Z" fill="#E6F0EC" stroke="#0F766E" stroke-width="1.5"/>
            <path d="M20 10v9m-4.5-4.5h9" stroke="#0F766E" stroke-width="2.4" stroke-linecap="round"/>
            <circle cx="20" cy="25.5" r="2.2" fill="#4D7CFE"/>
          </svg>
          <span>Hospital Intelligence System</span>
        </div>
        <div class="auth-header-actions">
          <a href="mailto:support@bagmaticare.local">Help &amp; support</a>
          <span class="workspace-status"><span aria-hidden="true"></span>Secure workspace</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(key="auth_shell", border=False):
        hero_col, form_col = st.columns([56, 44], gap=None, vertical_alignment="center")

        # ── HERO ──
        with hero_col.container(key="hero_panel", border=False):
            st.markdown("""
            <div class="hero-inner">
              <div class="hero-copy-block">
                <div class="hero-kicker">Clinical operations workspace</div>
                <div class="hero-headline">Hospital operations,<br>brought into focus.</div>
                <div class="hero-copy">See patient flow, capacity and workforce needs in one secure planning workspace.</div>
              </div>
              <div class="care-visual" aria-label="Abstract care pathway illustration">
                <svg viewBox="0 0 660 330" role="img">
                  <defs>
                    <linearGradient id="careSage" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#EFF5F1"/><stop offset="1" stop-color="#DDEBE6"/></linearGradient>
                    <filter id="careShadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#173F3B" flood-opacity=".08"/></filter>
                  </defs>
                  <path d="M48 247C111 196 145 215 198 168S285 87 350 130s76 114 139 82 72-99 126-111" fill="none" stroke="#B9D2C9" stroke-width="42" stroke-linecap="round" opacity=".55"/>
                  <path d="M48 247C111 196 145 215 198 168S285 87 350 130s76 114 139 82 72-99 126-111" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round" stroke-dasharray="5 9"/>
                  <g filter="url(#careShadow)">
                    <rect x="70" y="185" width="132" height="86" rx="16" fill="white" stroke="#D8E5E0"/>
                    <path d="M95 238v-27h22v27m-11-36v45m-22-22h44" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round"/>
                    <rect x="258" y="74" width="152" height="112" rx="18" fill="url(#careSage)" stroke="#CADFD7"/>
                    <path d="M292 155v-42h84v42M315 113V91h38v22M304 130h10m38 0h10m-58 14h10m38 0h10" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round"/>
                    <rect x="474" y="171" width="116" height="78" rx="16" fill="white" stroke="#D8E5E0"/>
                    <circle cx="516" cy="204" r="12" fill="#E6F0EC"/><path d="M516 197v14m-7-7h14" stroke="#0F766E" stroke-width="2" stroke-linecap="round"/>
                    <path d="M539 200h28m-28 10h18m-18 10h24" stroke="#9AB7AE" stroke-width="2" stroke-linecap="round"/>
                  </g>
                  <circle cx="48" cy="247" r="7" fill="#4D7CFE"/><circle cx="198" cy="168" r="7" fill="#0F766E"/><circle cx="350" cy="130" r="7" fill="#4D7CFE"/><circle cx="489" cy="212" r="7" fill="#0F766E"/><circle cx="615" cy="101" r="7" fill="#4D7CFE"/>
                </svg>
              </div>
              <div class="hero-trust"><span aria-hidden="true"></span>Designed for secure hospital planning.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── FORM PANEL ──
        with form_col.container(key="auth_region", border=False):
            with st.container(key="auth_card", border=False):

                _auth_title = "Welcome back" if view == "signin" else "Create your account"
                _auth_sub = ("Sign in to your secure hospital planning workspace."
                             if view == "signin" else
                             "Create secure access. You stay in control of what health information is saved.")
                st.markdown(f"""
                <div class="auth-view-marker view-{view}" aria-hidden="true"></div>
                <div class="auth-eyebrow">Secure patient access</div>
                <div class="auth-title">{_auth_title}</div>
                <div class="auth-subtitle">{_auth_sub}</div>
                """, unsafe_allow_html=True)

                # ── Segmented control ───────────────────────────────────────
                with st.container(key="seg_track", border=False):
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        if st.button("Sign In", use_container_width=True,
                                     type="primary" if view == "signin" else "secondary",
                                     key="tab_btn_signin"):
                            st.session_state.login_view = "signin"
                            st.session_state.reg_step = 1
                            st.rerun()
                    with tc2:
                        if st.button("Create Account", use_container_width=True,
                                     type="primary" if view == "register" else "secondary",
                                     key="tab_btn_reg"):
                            st.session_state.login_view = "register"
                            st.session_state.reg_step = 1
                            st.rerun()

                # ── Sign In view ─────────────────────────────────────────────
                if view == "signin":
                    if st.session_state.reg_success:
                        name = st.session_state.reg_success
                        st.markdown(f"""
                        <div class="success-banner">
                          <span aria-hidden="true" style="font-size:18px;">★</span>
                          <div><strong>Account created!</strong> Welcome, {name}. Sign in below.</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.reg_success = ""

                    if st.session_state.get("_sso_error"):
                        st.error(st.session_state.pop("_sso_error"))

                    if config.GOOGLE_CLIENT_ID:
                        _g_state = secrets.token_urlsafe(12)
                        st.session_state["_oauth_state"] = _g_state
                        _g_url = get_google_auth_url(_g_state)
                        st.markdown(
                            f'<a href="{_g_url}" target="_self" class="google-btn">'
                            f'<svg width="18" height="18" viewBox="0 0 48 48">'
                            f'<path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>'
                            f'<path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>'
                            f'<path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>'
                            f'<path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>'
                            f'</svg>'
                            f'Continue with Google'
                            f'</a>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="auth-divider"><div class="line"></div>'
                            '<span>or continue with credentials</span><div class="line"></div></div>',
                            unsafe_allow_html=True,
                        )

                    with st.form("login_form", clear_on_submit=False):
                        username = st.text_input("Username", placeholder="Enter your username",
                                                 icon=":material/person:", autocomplete="username",
                                                 key="login_username")
                        password = st.text_input("Password", type="password",
                                                 placeholder="Enter your password",
                                                 icon=":material/lock:", autocomplete="current-password",
                                                 key="login_password")
                        with st.container(key="login_options", border=False):
                            opt_remember, opt_forgot = st.columns([1, 1])
                            with opt_remember:
                                st.checkbox("Remember me", key="remember_login")
                            with opt_forgot:
                                st.markdown('<div style="text-align:right;padding-top:9px;">'
                                            '<a href="?forgot=1" class="auth-link">Forgot password?</a></div>',
                                            unsafe_allow_html=True)
                        submitted = st.form_submit_button("Sign in securely", use_container_width=True,
                                                          type="primary", icon=":material/login:")

                    if st.query_params.get("forgot") == "1":
                        st.caption("Password recovery is managed by your workspace administrator.")

                    if submitted:
                        if not username or not password:
                            st.error("Please enter both username and password.")
                        else:
                            with st.spinner("Signing you in…"):
                                role = verify_user(username, password)
                            if role:
                                st.session_state.logged_in  = True
                                st.session_state.username   = username
                                st.session_state.role       = role
                                st.session_state.login_view = "signin"
                                _save_session(username, role)
                                st.rerun()
                            else:
                                st.error("Incorrect username or password. Please try again.")

                    st.markdown("""
                    <div class="security-note">
                      Health information is saved only when you explicitly consent.
                    </div>
                    """, unsafe_allow_html=True)

                # ── Register view ────────────────────────────────────────────
                else:
                    step = st.session_state.reg_step
                    st.markdown(
                        f'<div class="register-step step-{"one" if step == 1 else "two"}"><div class="auth-section-label">Step {step} of 2 &nbsp;·&nbsp; '
                        f'{"Personal and account information" if step == 1 else "Password, security and consent"}</div></div>',
                        unsafe_allow_html=True)
                    if step == 1:
                        full_name = st.text_input("Full Name", placeholder="Enter your full name",
                                                  icon=":material/badge:", autocomplete="name",
                                                  key="reg_full_name")
                        new_user = st.text_input("Username", placeholder="Choose a username (min. 3 characters)",
                                                 icon=":material/person:", autocomplete="username",
                                                 key="reg_username")

                        rf1, rf2 = st.columns(2)
                        with rf1:
                            import datetime as _dt
                            st.date_input("Date of Birth", value=None,
                                          min_value=_dt.date(1900, 1, 1), max_value=_dt.date.today(),
                                          format="YYYY-MM-DD", key="reg_dob")
                        with rf2:
                            st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Non-binary"],
                                         key="reg_gender")

                        rf3, rf4 = st.columns(2)
                        with rf3:
                            st.text_input("Phone", placeholder="+977 98XXXXXXXX",
                                          icon=":material/call:", autocomplete="tel", key="reg_phone")
                        with rf4:
                            st.text_input("Email", placeholder="you@example.com",
                                          icon=":material/mail:", autocomplete="email", key="reg_email")

                        st.selectbox("District / City",
                                     ["— Select —", "Kathmandu", "Lalitpur", "Bhaktapur", "Kavrepalanchok",
                                      "Sindhupalchok", "Nuwakot", "Rasuwa", "Dhading", "Makwanpur",
                                      "Chitwan", "Sindhuli", "Ramechhap", "Dolakha", "Other"],
                                     key="reg_district")
                        st.caption("Date of birth, gender, phone, email and district are optional — add or update them any time from your profile.")

                        if st.button("Continue to security", type="primary", use_container_width=True,
                                     icon=":material/arrow_forward:", key="reg_continue"):
                            if not full_name.strip():
                                st.error("Full name is required.")
                            elif len(new_user.strip()) < 3:
                                st.error("Username must be at least 3 characters.")
                            else:
                                # Copied to non-widget keys: values read from a widget's own
                                # key can silently revert to blank once that widget stops
                                # being rendered and a later st.form_submit_button fires.
                                st.session_state.reg_committed_full_name = full_name
                                st.session_state.reg_committed_username = new_user
                                _dob_val = st.session_state.reg_dob
                                st.session_state.reg_committed_dob = _dob_val.isoformat() if _dob_val else ""
                                st.session_state.reg_committed_gender = st.session_state.reg_gender
                                st.session_state.reg_committed_phone = st.session_state.reg_phone
                                st.session_state.reg_committed_email = st.session_state.reg_email
                                st.session_state.reg_committed_district = st.session_state.reg_district
                                st.session_state.reg_step = 2
                                st.rerun()
                        reg_btn = False
                        new_pass = confirm_pass = ""
                        consent = False
                    else:
                        full_name = st.session_state.get("reg_committed_full_name", "")
                        new_user = st.session_state.get("reg_committed_username", "")
                        reg_dob = st.session_state.get("reg_committed_dob", "")
                        reg_gender = st.session_state.get("reg_committed_gender", "Prefer not to say")
                        reg_phone = st.session_state.get("reg_committed_phone", "")
                        reg_email = st.session_state.get("reg_committed_email", "")
                        reg_district = st.session_state.get("reg_committed_district", "— Select —")
                        rp1, rp2 = st.columns(2)
                        with rp1:
                            new_pass = st.text_input("Password", type="password",
                                                     placeholder="e.g. Nepal@2024!",
                                                     icon=":material/lock:", autocomplete="new-password",
                                                     key="reg_password")
                        with rp2:
                            confirm_pass = st.text_input("Confirm Password", type="password",
                                                         placeholder="Repeat password",
                                                         icon=":material/lock:", autocomplete="new-password",
                                                         key="reg_confirm")
                        if new_pass:
                            password_rules_card(new_pass)
                        with st.form("register_form", clear_on_submit=False):
                            consent = st.checkbox(
                                "I understand that after an estimate I can choose whether to save the "
                                "health information I entered. If I choose No, it will not be saved.",
                                key="reg_consent")
                            with st.container(key="reg_actions", border=False):
                                back_col, submit_col = st.columns([1, 2])
                                with back_col:
                                    back_btn = st.form_submit_button("Back", use_container_width=True)
                                with submit_col:
                                    reg_btn = st.form_submit_button(
                                        "Create account", use_container_width=True,
                                        type="primary", icon=":material/person_add:")
                        if back_btn:
                            st.session_state.reg_step = 1
                            st.rerun()

                    if reg_btn:
                        errors = []
                        if not full_name.strip():        errors.append("Full name is required.")
                        if len(new_user.strip()) < 3:    errors.append("Username must be at least 3 characters.")
                        if new_pass != confirm_pass:     errors.append("Passwords do not match.")
                        if not consent:                  errors.append("Please confirm that you understand how saving consent works.")
                        pw_ok, pw_errs = validate_password(new_pass)
                        if not pw_ok:
                            errors.extend(pw_errs)
                        if errors:
                            for e in errors:
                                st.error(e)
                        else:
                            with st.spinner("Creating your account…"):
                                ok, msg = register_user(
                                    new_user.strip(), new_pass, full_name=full_name.strip(),
                                    dob=reg_dob.strip(), gender=reg_gender,
                                    phone=reg_phone.strip(), email=reg_email.strip(),
                                    district=reg_district if reg_district != "— Select —" else "",
                                )
                            if ok:
                                st.session_state.reg_success = full_name.strip().split()[0]
                                st.session_state.login_view  = "signin"
                                st.session_state.reg_step = 1
                                st.rerun()
                            else:
                                st.error(msg)

                    st.markdown("""
                    <div class="security-note">
                      <span aria-hidden="true">✓</span>
                      You decide whether entered health information is saved after each estimate.
                    </div>
                    """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# PATIENT DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def _avatar_html(username: str, size: int, t: dict, name: str = "") -> str:
    """Return <img> or initials <div>. Thumbnail is resized to ~5 KB — safe to embed in HTML."""
    b64 = get_avatar_thumb_b64(username, size)
    r   = f"{size}px"
    common = (f"width:{r};height:{r};border-radius:50%;object-fit:cover;"
              f"border:3px solid rgba(255,255,255,.3);flex-shrink:0;")
    if b64:
        return f'<img src="data:image/jpeg;base64,{b64}" style="{common}">'
    # Fallback: initials
    label    = name or username
    initials = "".join(p[0].upper() for p in label.split()[:2]) or label[:2].upper()
    fsize    = max(12, size // 3)
    return (f'<div style="{common}background:linear-gradient(135deg,{t["secondary"]},{t["p_dark"]});'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-family:\'Plus Jakarta Sans\',sans-serif;'
            f'font-size:{fsize}px;font-weight:800;color:#fff;">{initials}</div>')


def _age_from_dob(dob_str: str):
    """Return integer age from a YYYY-MM-DD string, clamped 1-95. None if unparseable."""
    if not dob_str:
        return None
    try:
        from datetime import date
        dob   = date.fromisoformat(dob_str.strip())
        today = date.today()
        age   = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return max(1, min(95, age))
    except Exception:
        return None


@st.dialog("📅  Book Your Visit", width="large")
def _booking_dialog(est: dict, profile: dict, t: dict):
    import datetime as _dt
    uname = st.session_state.username

    hosp_name  = est.get("selected_hosp_name", "—")
    dept       = est.get("department", "—")

    # Block duplicate active booking for the same department
    _existing = [
        b for b in list_patient_bookings(uname)
        if b.get("department") == dept and b["status"] in ("pending", "confirmed")
    ]
    if _existing:
        _ex = _existing[0]
        st.markdown(
            f'<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:14px;'
            f'padding:20px 24px;text-align:center;">'
            f'<div style="font-size:2rem;margin-bottom:8px;">🚫</div>'
            f'<div style="font-weight:800;font-size:1rem;color:#991B1B;margin-bottom:6px;">'
            f'Already have an active booking for {dept}</div>'
            f'<div style="font-size:.82rem;color:#B91C1C;margin-bottom:12px;">'
            f'Ref <strong>{_ex["booking_ref"]}</strong> at <strong>{_ex["hospital_name"]}</strong> '
            f'is currently <strong>{_ex["status"]}</strong>.<br>'
            f'Cancel or wait for it to complete before booking the same department again.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return
    severity   = est.get("severity", "—")
    adm_type   = est.get("admission_type", "—")
    los_low    = int(est.get("los_low", 0))
    los_high   = int(est.get("los_high", 0))
    cost_low   = int(est.get("cost_low", 0))
    cost_high  = int(est.get("cost_high", 0))

    # ── Hero summary card ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(130deg,#1E1B4B,#4F46E5);'
        f'border-radius:16px;padding:20px 24px;margin-bottom:20px;">'
        f'<div style="font-size:.68rem;color:rgba(255,255,255,.55);letter-spacing:.1em;'
        f'text-transform:uppercase;margin-bottom:6px;">You are booking at</div>'
        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.25rem;'
        f'font-weight:800;color:#fff;margin-bottom:2px;">{hosp_name}</div>'
        f'<div style="font-size:.85rem;color:rgba(255,255,255,.7);margin-bottom:16px;">'
        f'{dept} &nbsp;·&nbsp; {severity} severity &nbsp;·&nbsp; {adm_type}</div>'
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
        f'<div style="background:rgba(255,255,255,.12);border-radius:12px;padding:10px 18px;'
        f'text-align:center;min-width:120px;">'
        f'<div style="font-size:.65rem;color:rgba(255,255,255,.55);text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:4px;">Est. Stay</div>'
        f'<div style="font-size:1.2rem;font-weight:800;color:#fff;">{los_low}–{los_high}</div>'
        f'<div style="font-size:.65rem;color:rgba(255,255,255,.55);">nights</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,.12);border-radius:12px;padding:10px 18px;'
        f'text-align:center;min-width:140px;">'
        f'<div style="font-size:.65rem;color:rgba(255,255,255,.55);text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:4px;">Est. Cost</div>'
        f'<div style="font-size:1.2rem;font-weight:800;color:#fff;">'
        f'Rs. {cost_low:,}–{cost_high:,}</div>'
        f'<div style="font-size:.65rem;color:rgba(255,255,255,.55);">rough estimate</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Section: When ────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-weight:700;font-size:.82rem;color:#374151;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">📆 When</div>',
        unsafe_allow_html=True,
    )
    d_col, t_col = st.columns(2)
    with d_col:
        min_date = _dt.date.today() + _dt.timedelta(days=1)
        req_date = st.date_input("Preferred Date", value=min_date,
                                 min_value=min_date, key="bk_date",
                                 label_visibility="collapsed")
    with t_col:
        pref_time = st.selectbox("Preferred Time Slot", ["Morning (8am–12pm)",
                                 "Afternoon (12pm–5pm)", "Any time"],
                                 key="bk_time", label_visibility="collapsed")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Section: Contact ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-weight:700;font-size:.82rem;color:#374151;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">📞 Contact</div>',
        unsafe_allow_html=True,
    )
    p_col, n_col = st.columns([1, 1.4])
    with p_col:
        phone = st.text_input("Phone Number", value=profile.get("phone", ""),
                              key="bk_phone", placeholder="98XXXXXXXX")
    with n_col:
        name_display = profile.get("full_name", "") or uname
        st.text_input("Full Name", value=name_display, key="bk_name_disp",
                      disabled=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Section: Notes ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-weight:700;font-size:.82rem;color:#374151;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">📝 Notes for Hospital</div>',
        unsafe_allow_html=True,
    )
    notes = st.text_area("Notes", key="bk_notes", height=80, label_visibility="collapsed",
                         placeholder="e.g. need wheelchair access, travelling from another district, previous surgery…")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
        'padding:10px 14px;margin-top:8px;display:flex;align-items:flex-start;gap:8px;">'
        '<span style="font-size:1rem;flex-shrink:0;">ℹ️</span>'
        '<div style="font-size:.76rem;color:#64748B;line-height:1.55;">'
        'This is a <strong>visit request</strong>, not a guaranteed slot. '
        'The hospital admin will review and confirm. '
        'You will be notified via the notification bell.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Actions ──────────────────────────────────────────────────────────────
    b_col, c_col = st.columns([3, 1])
    with b_col:
        if st.button("✅  Confirm Booking Request", type="primary",
                     use_container_width=True, key="bk_submit"):
            if not phone.strip():
                st.error("Please enter a contact phone number.")
            else:
                ok, ref = submit_booking({
                    "patient_username": uname,
                    "patient_id":       get_patient_id(uname),
                    "full_name":        profile.get("full_name", ""),
                    "phone":            phone.strip(),
                    "hospital_id":      est.get("hospital_id", ""),
                    "hospital_name":    hosp_name,
                    "department":       dept,
                    "severity":         severity,
                    "admission_type":   adm_type,
                    "requested_date":   str(req_date),
                    "preferred_time":   pref_time,
                    "notes":            notes.strip(),
                    "los_low":          est.get("los_low"),
                    "los_high":         est.get("los_high"),
                    "cost_low":         est.get("cost_low"),
                    "cost_high":        est.get("cost_high"),
                })
                if ok:
                    st.session_state["_last_booking_ref"] = ref
                    st.session_state["_booking_done"] = True
                    st.rerun()
                else:
                    st.error(f"Booking failed: {ref}")
    with c_col:
        if st.button("Cancel", use_container_width=True, key="bk_cancel"):
            st.rerun()



@st.dialog("Can we save your estimate?", width="small")
def _consent_popup(data: dict, t: dict):
    st.markdown(
        f'<div style="text-align:center;padding:6px 0 14px;">'
        f'<div style="font-size:2.2rem;margin-bottom:8px;">🔒</div>'
        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.95rem;'
        f'font-weight:800;color:{t["t1"]};margin-bottom:6px;">Help us improve estimates</div>'
        f'<div style="font-size:.8rem;color:{t["t2"]};line-height:1.7;margin-bottom:14px;">'
        f'We would like to save the details you just filled in — your age, condition, '
        f'department, and hospital choice — to help make future estimates more accurate '
        f'for everyone in the Bagmati region.'
        f'</div>'
        f'<div style="background:{t["p_light"]};border-radius:10px;padding:11px 14px;'
        f'font-size:.75rem;color:{t["t2"]};text-align:left;line-height:1.6;margin-bottom:16px;">'
        f'<strong>What gets saved:</strong><br>'
        f'Age · Gender · Severity · Admission type · Department · Hospital · '
        f'Chronic condition (yes/no) · No. of chronic conditions · Insurance (yes/no) · '
        f'Visit purpose · Travel distance · District · '
        f'Hospital occupancy at time of estimate · Times you have used the planner'
        f'<br><br>'
        f'<strong>What does NOT get saved:</strong><br>'
        f'Your name, phone, address, diagnosis text, or any personal health records.'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("We'll remember your choice for the rest of this session — you won't be asked again.")
    c1, c2 = st.columns(2, gap="small")
    if c1.button("Yes, save it", type="primary", use_container_width=True):
        st.session_state["_consent_choice"] = True
        st.session_state["_est_saved"] = save_estimate_request(data)
        st.session_state["_est_awaiting_consent"] = False
        st.session_state["_est_loading"] = True
        st.rerun()
    if c2.button("No thanks", use_container_width=True):
        st.session_state["_consent_choice"] = False
        st.session_state["_est_saved"] = False
        st.session_state["_est_awaiting_consent"] = False
        st.session_state["_est_loading"] = True
        st.rerun()


@st.dialog("Hospital Details", width="large")
def _hospital_popup(hid: str, occ_h, adm, hosp_df, dept_map, t: dict):
    hname  = config.HOSPITAL_NAMES.get(hid, hid)
    hloc   = config.HOSPITAL_LOCATIONS.get(hid, "")
    hlvl   = config.HOSPITAL_LEVELS.get(hid, "")
    htype  = config.HOSPITAL_TYPES.get(hid, "")
    hbeds  = config.HOSPITAL_BEDS.get(hid, 0)

    hrow = (hosp_df[hosp_df["hospital_id"] == hid].iloc[0]
            if hid in hosp_df["hospital_id"].values else {})

    def _fv(v, default=0.0):
        return float(v) if pd.notna(v) else default

    occ_v     = _fv(occ_h[occ_h["hospital_id"] == hid]["occupancy_rate"].mean()
                    if "hospital_id" in occ_h.columns and hid in occ_h["hospital_id"].values
                    else 0.75, 0.75)
    beds_free = max(0, hbeds - int(occ_v * hbeds))
    rec_rate  = _fv(hrow.get("recovery_rate")   if hasattr(hrow, "get") else None)
    doc_count = int(_fv(hrow.get("doctor_count") if hasattr(hrow, "get") else None))
    avg_exp   = _fv(hrow.get("avg_exp")          if hasattr(hrow, "get") else None)
    wait_min  = _fv(hrow.get("avg_wait_triage")  if hasattr(hrow, "get") else None)

    if occ_v > 0.85:
        acc="#EF4444"; occ_lbl="Very busy right now"; occ_bg="#FEF2F2"
    elif occ_v > 0.75:
        acc="#F59E0B"; occ_lbl="Moderately busy";     occ_bg="#FFFBEB"
    else:
        acc="#10B981"; occ_lbl="Good availability";    occ_bg="#F0FDF4"

    lvl_color  = "#4F46E5" if hlvl == "Tertiary" else "#7C3AED"
    h_adm_all  = adm[adm["hospital_id"] == hid].copy()
    hosp_cost  = float(h_adm_all["total_bill_npr"].mean()) if len(h_adm_all) else 0
    rec_in_10  = round(rec_rate * 10)

    mnames = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["p_light"]};border-radius:14px;padding:14px 18px;margin-bottom:16px;">'
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">'
        f'<div>'
        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.1rem;'
        f'font-weight:800;color:{t["t1"]};">{hname}</div>'
        f'<div style="font-size:.72rem;color:{t["t3"]};margin-top:3px;">'
        f'📍 {hloc} &nbsp;·&nbsp; {htype} &nbsp;·&nbsp; {hbeds} beds</div>'
        f'</div>'
        f'<div style="display:flex;gap:6px;flex-shrink:0;align-items:center;">'
        f'<span style="background:#EEF2FF;color:{lvl_color};font-size:.6rem;font-weight:700;'
        f'padding:3px 9px;border-radius:9999px;">{hlvl}</span>'
        f'<span style="background:{occ_bg};color:{acc};font-size:.6rem;font-weight:700;'
        f'padding:3px 9px;border-radius:9999px;">{occ_lbl}</span>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── 4 plain-language fact boxes ──────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    for col, ico, question, answer, note in [
        (k1,"🛏️","Beds free right now",  f"{beds_free} of {hbeds}", f"{occ_v:.0%} currently in use"),
        (k2,"❤️","Patients go home well", f"{rec_in_10} out of 10",  "of everyone admitted here"),
        (k3,"💰","Typical visit costs",   f"Rs. {hosp_cost:,.0f}",   "average across all visits"),
        (k4,"⏱️","You'll be seen in",     f"~{wait_min:.0f} min",     "after you arrive"),
    ]:
        col.markdown(
            f'<div style="background:#fff;border:1px solid {t["border"]};border-radius:12px;'
            f'padding:13px 14px;text-align:center;height:100%;">'
            f'<div style="font-size:1.3rem;margin-bottom:6px;">{ico}</div>'
            f'<div style="font-size:.65rem;color:{t["t3"]};margin-bottom:4px;">{question}</div>'
            f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1rem;'
            f'font-weight:800;color:{t["t1"]};margin-bottom:3px;">{answer}</div>'
            f'<div style="font-size:.62rem;color:{t["t3"]}">{note}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Extra facts row ──────────────────────────────────────────────────────
    e1,e2,e3 = st.columns(3)
    for col, ico, lbl, val in [
        (e1,"👨‍⚕️","Doctors on staff",       f"{doc_count} doctors · avg {avg_exp:.0f} yrs experience"),
        (e2,"🏥","Departments available",    f"{len(dept_map.get(hid,[]))} specialties"),
        (e3,"🔬","Hospital type",            f"{htype} · {hlvl} level"),
    ]:
        col.markdown(
            f'<div style="background:{t["p_light"]};border-radius:10px;padding:11px 13px;">'
            f'<div style="font-size:.75rem;font-weight:700;color:{t["t1"]};margin-bottom:2px;">'
            f'{ico} {lbl}</div>'
            f'<div style="font-size:.72rem;color:{t["t2"]};">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Department filter + chart ────────────────────────────────────────────
    if "admission_month" in h_adm_all.columns and "admission_hour" in h_adm_all.columns:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        depts    = sorted(dept_map.get(hid, []))
        dept_sel = st.selectbox(
            "Filter by department",
            ["All departments"] + depts,
            key=f"dept_{hid}",
        )

        view = (h_adm_all if dept_sel == "All departments"
                else h_adm_all[h_adm_all["department_name"] == dept_sel])

        # Department fact strip — only when a dept is chosen
        if dept_sel != "All departments" and len(view):
            d_cost  = float(view["total_bill_npr"].mean())
            d_los   = float(view["length_of_stay_days"].median())
            d_rec10 = round((view["discharge_outcome"] == "Recovered").mean() * 10)
            f1,f2,f3 = st.columns(3)
            for col, ico, lbl, val in [
                (f1,"💰","Typical cost",    f"Rs. {d_cost:,.0f}"),
                (f2,"🛏️","Typical stay",    f"{d_los:.0f} nights"),
                (f3,"❤️","Go home well",    f"{d_rec10} out of 10"),
            ]:
                col.markdown(
                    f'<div style="background:{t["p_light"]};border-radius:10px;'
                    f'padding:10px 12px;text-align:center;margin:8px 0 4px;">'
                    f'<div style="font-size:.65rem;color:{t["t3"]};margin-bottom:2px;">{ico} {lbl}</div>'
                    f'<div style="font-size:.9rem;font-weight:800;color:{t["t1"]};">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if len(view) == 0:
            st.info("No data for this department.")
        else:
            # Best time cards — recalculate from filtered view
            hr_cnt = (view.groupby("admission_hour").size()
                      .reindex(range(24),fill_value=0).reset_index())
            hr_cnt.columns = ["hour","count"]
            quietest_h = int(hr_cnt.loc[hr_cnt["count"].idxmin(),"hour"])
            busiest_h  = int(hr_cnt.loc[hr_cnt["count"].idxmax(),"hour"])

            mo2 = view.groupby("admission_month").size().reset_index(name="n")
            mo2["mname"] = mo2["admission_month"].map(mnames)
            mo_min = int(mo2.loc[mo2["n"].idxmin(),"admission_month"])
            mo_max = int(mo2.loc[mo2["n"].idxmax(),"admission_month"])
            qm_lbl = mnames.get(mo_min,"—")
            bm_lbl = mnames.get(mo_max,"—")

            ctx = dept_sel if dept_sel != "All departments" else hname
            st.markdown(
                f'<div style="font-size:.75rem;font-weight:700;color:{t["t1"]};margin:10px 0 6px;">'
                f'📅 Best time to visit — {ctx}</div>',
                unsafe_allow_html=True,
            )

            b1,b2,b3 = st.columns(3)
            for col, ico, lbl, val, detail, bg, tc in [
                (b1,"🕐","Best time of day",  f"{quietest_h:02d}:00",
                 "Fewest patients arriving",  "#F0FDF4","#065F46"),
                (b2,"🗓️","Best month",         qm_lbl,
                 "Lowest volume of the year", "#F0FDF4","#065F46"),
                (b3,"⚠️","Try to avoid",       f"{busiest_h:02d}:00 · {bm_lbl}",
                 "Busiest period — longer wait", "#FEF2F2","#991B1B"),
            ]:
                col.markdown(
                    f'<div style="background:{bg};border-radius:12px;padding:11px 13px;">'
                    f'<div style="font-size:1.1rem;margin-bottom:3px;">{ico}</div>'
                    f'<div style="font-size:.62rem;color:#6B7280;margin-bottom:2px;">{lbl}</div>'
                    f'<div style="font-size:.85rem;font-weight:800;color:{tc};margin-bottom:1px;">{val}</div>'
                    f'<div style="font-size:.6rem;color:#6B7280;">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Two charts side by side
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            ch_left, ch_right = st.columns(2, gap="medium")

            # ── Left: smooth area line — monthly trend ───────────────────────
            with ch_left:
                st.markdown(
                    f'<div style="font-size:.72rem;font-weight:700;color:{t["t1"]};margin-bottom:4px;">'
                    f'📅 Busiest months</div>',
                    unsafe_allow_html=True,
                )
                fig_area = go.Figure()
                fig_area.add_trace(go.Scatter(
                    x=mo2["mname"], y=mo2["n"],
                    mode="lines", fill="tozeroy",
                    fillcolor="rgba(99,102,241,.10)",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_area.add_trace(go.Scatter(
                    x=mo2["mname"], y=mo2["n"],
                    mode="lines+markers",
                    line=dict(color=t["primary"], width=2.5, shape="spline", smoothing=0.8),
                    marker=dict(
                        color=[acc if v==mo2["n"].max()
                               else ("#10B981" if v==mo2["n"].min() else t["primary"])
                               for v in mo2["n"]],
                        size=[13 if v in (mo2["n"].max(), mo2["n"].min()) else 7
                              for v in mo2["n"]],
                        line=dict(color="#fff", width=2),
                    ),
                    hovertemplate="<b>%{x}</b><br>%{y:,} patient visits<extra></extra>",
                    showlegend=False,
                ))
                lay_a = chart_layout(t, 210)
                lay_a["margin"] = dict(l=30, r=10, t=4, b=28)
                lay_a["yaxis"]["tickformat"] = ","
                lay_a["yaxis"]["showgrid"] = True
                fig_area.update_layout(**lay_a)
                pchart(fig_area, key=f"area_{hid}_{dept_sel}")
                qm = mo2.loc[mo2["n"].idxmin(), "mname"]
                bm = mo2.loc[mo2["n"].idxmax(), "mname"]
                st.markdown(
                    f'<div style="font-size:.68rem;color:{t["t3"]};margin-top:2px;">'
                    f'<span style="color:#10B981;font-weight:700;">● {qm}</span> quietest &nbsp;'
                    f'<span style="color:{acc};font-weight:700;">● {bm}</span> busiest</div>',
                    unsafe_allow_html=True,
                )

            # ── Right: horizontal lollipop — day of week ─────────────────────
            with ch_right:
                st.markdown(
                    f'<div style="font-size:.72rem;font-weight:700;color:{t["t1"]};margin-bottom:4px;">'
                    f'📆 Busiest days of the week</div>',
                    unsafe_allow_html=True,
                )
                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                day_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                dow_cnt   = (view["admission_date"].dt.dayofweek
                             .map(dict(enumerate(day_short)))
                             .value_counts()
                             .reindex(day_short, fill_value=0)
                             .reset_index())
                dow_cnt.columns = ["day","n"]
                dow_sorted = dow_cnt.sort_values("n", ascending=True).reset_index(drop=True)

                dot_colors = [
                    "#10B981" if v == dow_sorted["n"].min()
                    else (acc   if v == dow_sorted["n"].max() else t["primary"])
                    for v in dow_sorted["n"]
                ]

                fig_lol = go.Figure()
                for i, row in dow_sorted.iterrows():
                    fig_lol.add_shape(
                        type="line", layer="below",
                        x0=0, x1=float(row["n"]), y0=i, y1=i,
                        line=dict(color=dot_colors[i], width=1.5, dash="dot"),
                    )
                fig_lol.add_trace(go.Scatter(
                    x=dow_sorted["n"], y=list(range(len(dow_sorted))),
                    mode="markers+text",
                    marker=dict(color=dot_colors, size=14,
                                line=dict(color="#fff", width=2)),
                    text=[f" {v:,}" for v in dow_sorted["n"]],
                    textposition="middle right",
                    textfont=dict(size=9, color=t["t2"]),
                    hovertemplate="<b>%{customdata}</b><br>%{x:,} patient visits<extra></extra>",
                    customdata=dow_sorted["day"],
                    showlegend=False,
                ))
                lay_l = chart_layout(t, 210)
                lay_l["yaxis"] = dict(
                    tickmode="array", tickvals=list(range(len(dow_sorted))),
                    ticktext=list(dow_sorted["day"]),
                    showgrid=False, zeroline=False,
                    linecolor="rgba(0,0,0,0)",
                    tickfont=dict(size=10, color=t["t3"]),
                )
                lay_l["xaxis"] = dict(showgrid=True, gridcolor="rgba(0,0,0,.04)",
                                      zeroline=False, linecolor="rgba(0,0,0,0)",
                                      tickfont=dict(size=8, color=t["t3"]),
                                      tickformat=",")
                lay_l["margin"] = dict(l=60, r=50, t=4, b=10)
                fig_lol.update_layout(**lay_l)
                pchart(fig_lol, key=f"lol_{hid}_{dept_sel}")
                qd = dow_sorted.loc[dow_sorted["n"].idxmin(), "day"]
                bd = dow_sorted.loc[dow_sorted["n"].idxmax(), "day"]
                st.markdown(
                    f'<div style="font-size:.68rem;color:{t["t3"]};margin-top:2px;">'
                    f'<span style="color:#10B981;font-weight:700;">● {qd}</span> quietest &nbsp;'
                    f'<span style="color:{acc};font-weight:700;">● {bd}</span> busiest</div>',
                    unsafe_allow_html=True,
                )


def build_patient_notifications(profile: dict, occ_h: pd.DataFrame,
                                adm: pd.DataFrame,
                                username: str = "") -> list:
    """Return list of notification dicts for the patient bell.
    Each dict: {severity, icon, title, body}
    severity: 'urgent' | 'warning' | 'info' | 'success'
    """
    notes = []
    now_month = pd.Timestamp.now().month

    # 0a. Booking status changes
    if username:
        from src.hospital_connector import list_patient_bookings as _lpb
        for bk in _lpb(username):
            if bk["status"] == "confirmed":
                notes.append({
                    "severity": "success",
                    "icon":     "📅",
                    "title":    f"Booking {bk['booking_ref']} confirmed!",
                    "body":     f"{bk['hospital_name']} · {bk['department']} · {bk['requested_date']}. Check My Bookings tab.",
                })
            elif bk["status"] == "cancelled":
                note_txt = f" Reason: {bk['admin_note']}" if bk.get("admin_note") else ""
                notes.append({
                    "severity": "warning",
                    "icon":     "❌",
                    "title":    f"Booking {bk['booking_ref']} was cancelled",
                    "body":     f"{bk['hospital_name']} · {bk['requested_date']}.{note_txt}",
                })

    # 0b. Unread admin chat messages
    if username:
        unread_msgs = count_unread_for_patient(username)
        if unread_msgs:
            notes.append({
                "severity": "urgent",
                "icon":     "💬",
                "title":    f"You have {unread_msgs} unread message{'s' if unread_msgs > 1 else ''} from admin",
                "body":     "Open the 💬 Chat tab to read and reply.",
            })

    # 1. Preferred hospital occupancy
    pref_hosp = profile.get("pref_hospital", "Any hospital") or "Any hospital"
    pref_id   = next((k for k, v in config.HOSPITAL_NAMES.items() if v == pref_hosp), None)
    if pref_id and "hospital_id" in occ_h.columns:
        occ_val  = float(occ_h[occ_h["hospital_id"] == pref_id]["occupancy_rate"].mean()
                         if pref_id in occ_h["hospital_id"].values else 0.75)
        hbeds    = config.HOSPITAL_BEDS.get(pref_id, 0)
        beds_free = max(0, hbeds - int(occ_val * hbeds))
        if occ_val > 0.85:
            notes.append({"severity": "urgent", "icon": "🔴",
                          "title": f"{pref_hosp} is very busy right now",
                          "body": f"Only {beds_free} beds free ({occ_val:.0%} occupied). "
                                  "Consider visiting another hospital or waiting a few days."})
        elif occ_val > 0.75:
            notes.append({"severity": "warning", "icon": "🟡",
                          "title": f"{pref_hosp} is moderately busy",
                          "body": f"{beds_free} beds available ({occ_val:.0%} occupied). "
                                  "It's worth calling ahead before you arrive."})
        else:
            notes.append({"severity": "success", "icon": "🟢",
                          "title": f"{pref_hosp} has good availability",
                          "body": f"{beds_free} of {hbeds} beds are currently free — a good time to visit."})

    # 2. Surge alert for any hospital at critical level (skip preferred, already shown)
    if "hospital_id" in occ_h.columns:
        for hid, hname in config.HOSPITAL_NAMES.items():
            if hid == pref_id:
                continue
            occ_val = float(occ_h[occ_h["hospital_id"] == hid]["occupancy_rate"].mean()
                            if hid in occ_h["hospital_id"].values else 0.75)
            if occ_val > 0.88:
                notes.append({"severity": "warning", "icon": "⚠️",
                              "title": f"{hname} is near capacity",
                              "body": f"This hospital is {occ_val:.0%} full. "
                                      "Choose a different hospital if you have flexibility."})

    # 3. Monthly tip — busier-than-average month?
    month_avg = adm.groupby("admission_month")["length_of_stay_days"].mean()
    overall_avg = adm["length_of_stay_days"].mean()
    if now_month in month_avg.index:
        month_los = month_avg[now_month]
        month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        mname = month_names.get(now_month, "This month")
        if month_los > overall_avg * 1.08:
            notes.append({"severity": "info", "icon": "📅",
                          "title": f"{mname} is a busy month",
                          "body": "Hospitals tend to have longer stays and more patients this month. "
                                  "Non-urgent visits are better scheduled next month."})
        else:
            notes.append({"severity": "success", "icon": "📅",
                          "title": f"{mname} is a quiet month to visit",
                          "body": "Patient numbers are lower than average — expect shorter queues "
                                  "and more staff attention."})

    # 4. Profile completeness
    missing = [f.replace("_", " ") for f in ["full_name", "dob", "gender", "phone"]
               if not (profile.get(f) or "").strip()]
    if missing:
        notes.append({"severity": "info", "icon": "👤",
                      "title": "Your profile is incomplete",
                      "body": f"Add your {', '.join(missing)} to get more accurate estimates "
                              "and save time at registration."})

    # 5. Emergency contact missing
    if not (profile.get("ec_name") or "").strip():
        notes.append({"severity": "info", "icon": "📞",
                      "title": "No emergency contact saved",
                      "body": "Hospitals always ask for an emergency contact. "
                              "Add one in My Profile so you're prepared."})

    return notes


def _notif_card_html(n: dict) -> str:
    """Render one notification as an HTML card string."""
    colors = {
        "urgent":  ("#FEE2E2", "#EF4444", "#991B1B"),
        "warning": ("#FEF3C7", "#F59E0B", "#92400E"),
        "success": ("#D1FAE5", "#10B981", "#065F46"),
        "info":    ("#EEF2FF", "#6366F1", "#3730A3"),
    }
    bg, border, text = colors.get(n["severity"], colors["info"])
    safe_title = str(n["title"]).replace("<", "&lt;").replace(">", "&gt;")
    safe_body  = str(n["body"]).replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
        f'padding:11px 14px;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:flex-start;gap:9px;">'
        f'<span style="font-size:1.1rem;flex-shrink:0;margin-top:1px;">{n["icon"]}</span>'
        f'<div>'
        f'<div style="font-weight:700;font-size:.82rem;color:{text};margin-bottom:3px;">{safe_title}</div>'
        f'<div style="font-size:.76rem;color:{text};opacity:.85;line-height:1.45;">{safe_body}</div>'
        f'</div></div></div>'
    )


def _chat_bubble_html(body: str, is_mine: bool, sent_at=None) -> str:
    align  = "flex-end"              if is_mine else "flex-start"
    bg     = "#4F46E5"               if is_mine else "#F3F4F6"
    color  = "#fff"                  if is_mine else "#111827"
    radius = "18px 18px 4px 18px"   if is_mine else "18px 18px 18px 4px"
    ta     = "right"                 if is_mine else "left"
    ts     = sent_at.strftime("%H:%M") if sent_at and hasattr(sent_at, "strftime") else ""
    safe_body = str(body).replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div style="display:flex;justify-content:{align};margin-bottom:10px;">'
        f'<div style="max-width:74%;background:{bg};color:{color};border-radius:{radius};'
        f'padding:10px 14px;font-size:.88rem;line-height:1.55;word-break:break-word;">'
        f'<div>{safe_body}</div>'
        f'<div style="font-size:.65rem;opacity:.55;margin-top:5px;text-align:{ta};">{ts}</div>'
        f'</div></div>'
    )


def patient_dashboard():
    t = PATIENT_THEME
    inject_global_css(t)
    # Patients navigate entirely through the "Choose what you want to do" tab bar
    # below — the sidebar's nav duplicated that, so it's hidden here.
    st.markdown('<style>[data-testid="stSidebar"] { display:none !important; }</style>',
                unsafe_allow_html=True)

    adm      = load_admissions()
    occ      = load_occupancy()
    occ_h    = load_occupancy_with_hospital()
    hosp_df  = hospital_stats()
    dept_map = departments_by_hospital()

    avg_los    = adm["length_of_stay_days"].mean()
    avg_cost   = adm["total_bill_npr"].mean()
    low_season = occ.groupby("season")["occupancy_rate"].mean().idxmin()
    low_occ    = occ.groupby("season")["occupancy_rate"].mean().min()

    # ── Load profile ──
    uname   = st.session_state.username
    profile = get_profile(uname)
    p_full  = profile.get("full_name") or uname

    profile_age     = _age_from_dob(profile.get("dob",""))
    profile_chronic = bool((profile.get("chronic_conditions") or "").strip())

    # ── Notifications (computed first so bell can show count) ────────────────
    notifs  = build_patient_notifications(profile, occ_h, adm, username=uname)
    n_total = len(notifs)

    # ── Session bar: [status] [🔔 bell] [Sign Out] ───────────────────────────
    render_session_bar(notifs=notifs, n_total=n_total)

    # ── Welcome banner (full-width, no bell column) ──────────────────────────
    rec_rate      = float((adm["discharge_outcome"] == "Recovered").mean())
    n_hospitals   = len(config.HOSPITAL_NAMES)
    pid_banner    = get_patient_id(uname)
    joined_banner = get_user_created_at(uname)

    st.markdown(welcome_banner(
        f"Welcome back, {p_full.split()[0]}",
        "Plan your hospital visit without needing any medical knowledge — Bagmati Region, Nepal",
        t,
        patient_id = pid_banner,
        joined     = joined_banner,
    ), unsafe_allow_html=True)

    # A calm, useful landing summary before patients choose a workflow.
    st.markdown(
        f'''<section class="patient-overview" aria-labelledby="patient-overview-title">
          <div class="patient-overview-head">
            <div>
              <div class="patient-overview-kicker">At a glance</div>
              <div class="patient-overview-title" id="patient-overview-title">Plan with confidence</div>
            </div>
            <div class="patient-overview-note">Based on anonymised Bagmati hospital records</div>
          </div>
          <div class="patient-overview-grid">
            <div class="patient-stat">
              <div class="patient-stat-top"><span class="patient-stat-icon" aria-hidden="true">&#127973;</span></div>
              <div class="patient-stat-value">{n_hospitals}</div>
              <div class="patient-stat-label">Hospitals to compare</div>
              <div class="patient-stat-sub">Review capacity, departments and costs.</div>
            </div>
            <div class="patient-stat">
              <div class="patient-stat-top"><span class="patient-stat-icon" aria-hidden="true">&#128719;</span></div>
              <div class="patient-stat-value">{avg_los:.1f} days</div>
              <div class="patient-stat-label">Typical hospital stay</div>
              <div class="patient-stat-sub">Regional average across recorded visits.</div>
            </div>
            <div class="patient-stat">
              <div class="patient-stat-top"><span class="patient-stat-icon" aria-hidden="true">&#8360;</span></div>
              <div class="patient-stat-value">Rs. {avg_cost:,.0f}</div>
              <div class="patient-stat-label">Typical admission cost</div>
              <div class="patient-stat-sub">Use My Estimate for a personal range.</div>
            </div>
            <div class="patient-stat">
              <div class="patient-stat-top"><span class="patient-stat-icon" aria-hidden="true">&#9728;</span></div>
              <div class="patient-stat-value">{low_season}</div>
              <div class="patient-stat-label">Quieter season</div>
              <div class="patient-stat-sub">Average occupancy is {low_occ:.0%} in this period.</div>
            </div>
          </div>
        </section>''',
        unsafe_allow_html=True,
    )

    _unread_chat = count_unread_for_patient(uname)
    _chat_label  = f"💬  Chat {'🔴' if _unread_chat else ''}"

    # ── Toast notification for new admin reply ────────────────────────────────
    if "_pat_unread_seen" not in st.session_state:
        st.session_state._pat_unread_seen = _unread_chat
    elif _unread_chat > st.session_state._pat_unread_seen:
        st.toast("💬 New message from admin!", icon="🔔")
        st.session_state._pat_unread_seen = _unread_chat

    _pending_bk = sum(1 for b in list_patient_bookings(uname) if b["status"] == "confirmed")
    _bk_label   = f"📅  My Bookings {'🟢' if _pending_bk else ''}"

    st.markdown('<div class="dashboard-nav-label">Choose what you want to do</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "💰  Estimate",
        "📊  Care insights",
        "📅  Visit timing",
        "👴  Cost by age",
        "🏥  Hospitals",
        _bk_label,
        _chat_label,
        "👤  Profile",
    ])

    # ════ TAB 1: PLANNER ════════════════════════════════════════════════════
    with tab1:
        if not models_ready():
            st.warning("⚠️ Models not trained. Run `python src/train_models.py` first.")
            return

        # Explainer
        st.markdown(f"""
        <div style="background:{t['p_light']};border:1px solid {t['border']};border-radius:14px;
          padding:16px 20px;display:flex;align-items:flex-start;gap:12px;margin-bottom:20px;">
          <span style="font-size:1.5rem;">🎯</span>
          <div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
              font-size:.92rem;color:{t['t1']};margin-bottom:3px;">
              Find out how long you might stay and what it could cost
            </div>
            <div style="font-size:.82rem;color:{t['t2']};line-height:1.6;">
              Fill in a few details below and we'll give you a personalised estimate based on
              <strong>120,000+ real hospital visits</strong> in the Bagmati region. No medical knowledge needed.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_form, col_result = st.columns([1, 1.1], gap="large")

        with col_form:
            # Profile-fill status banner
            filled_fields = []
            if profile_age:     filled_fields.append("age")
            if profile_chronic: filled_fields.append("chronic condition")
            if filled_fields:
                st.markdown(f"""
                <div style="background:#EEF2FF;border:1px solid rgba(70,72,212,.18);border-radius:10px;
                  padding:10px 14px;font-size:.79rem;color:#3730a3;margin-bottom:14px;
                  display:flex;align-items:center;gap:9px;">
                  <span style="font-size:1.1rem;">&#10024;</span>
                  <span><strong>Pre-filled from your profile:</strong>
                  {', '.join(filled_fields)}.
                  You can still adjust any value below.</span>
                </div>
                """, unsafe_allow_html=True)

            def _field_label(txt, help_txt=""):
                st.markdown(
                    f'<div style="font-size:.8rem;font-weight:600;color:{t["t2"]};'
                    f'margin-bottom:5px;margin-top:10px;">{txt}</div>',
                    unsafe_allow_html=True,
                )

            # ── Age ──────────────────────────────────────────────────────────
            age_default = profile_age if profile_age else 35
            age = st.slider("Your Age", 1, 95, age_default,
                            help="Drag to set your age. We use this to give a more accurate estimate.")
            if profile_age:
                st.markdown(
                    f'<div style="font-size:.72rem;color:#3730a3;margin-top:-10px;margin-bottom:4px;">'
                    f'✨ Auto-filled from your date of birth — adjust if needed</div>',
                    unsafe_allow_html=True,
                )

            # ── Gender ───────────────────────────────────────────────────────
            _field_label("Your gender")
            gender = st.radio(
                "Your gender", ["Male", "Female", "Prefer not to say"],
                horizontal=True, label_visibility="collapsed",
                key="est_gender",
            )

            # ── Severity ─────────────────────────────────────────────────────
            _field_label("How serious is your condition?")
            severity = st.selectbox(
                "severity", config.SEVERITY_OPTIONS,
                label_visibility="collapsed",
                help="Mild = can walk around · Moderate = needs regular care · Severe / Critical = intensive care",
            )

            # ── Admission type ───────────────────────────────────────────────
            _field_label("How are you coming in?")
            admission_type = st.selectbox(
                "admission_type", config.ADMISSION_TYPE_OPTIONS,
                label_visibility="collapsed",
                help="Emergency = urgent · Elective = planned · Referral = sent by another doctor",
            )

            # ── Department ───────────────────────────────────────────────────
            _field_label("Which part of the hospital?")
            department = st.selectbox(
                "department", config.DEPARTMENT_OPTIONS,
                label_visibility="collapsed",
                help="Not sure? Pick the one closest to your health issue",
            )

            # ── Who is this for ──────────────────────────────────────────────
            _field_label("Who is this visit for?")
            visit_for = st.radio(
                "Who is this visit for?",
                ["Myself", "A family member", "A child"],
                horizontal=True, label_visibility="collapsed",
                key="est_visit_for",
            )

            # ── Chronic condition ────────────────────────────────────────────
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            has_chronic = st.toggle(
                "I have a long-term health condition",
                value=profile_chronic,
                help="e.g. diabetes, high blood pressure, asthma, heart disease",
            )
            if profile_chronic:
                st.markdown(
                    f'<div style="font-size:.72rem;color:#4F46E5;margin-top:-8px;margin-bottom:4px;">'
                    f'✨ Filled from your health profile — '
                    f'<em>{(profile.get("chronic_conditions") or "")[:40]}'
                    f'{"…" if len(profile.get("chronic_conditions",""))>40 else ""}</em></div>',
                    unsafe_allow_html=True,
                )
            num_chronic = 0
            if has_chronic:
                num_chronic = st.number_input(
                    "How many chronic conditions?",
                    min_value=1, max_value=10, value=1, step=1,
                    help="Count each separate condition — e.g. diabetes + hypertension = 2",
                )

            # ── Insurance ────────────────────────────────────────────────────
            has_insurance = st.toggle(
                "I have health insurance",
                value=False,
                help="Having insurance may reduce your out-of-pocket cost. We'll show what to check.",
            )

            # ── Travelling far? ──────────────────────────────────────────────
            _field_label("How far are you travelling?")
            travel_dist = st.radio(
                "travel_dist",
                ["Within the city", "More than 1 hour away"],
                horizontal=True, label_visibility="collapsed",
                key="est_travel",
            )

            # ── Hospital ─────────────────────────────────────────────────────
            _field_label("Which hospital are you visiting?")
            hosp_display = {v: k for k, v in config.HOSPITAL_NAMES.items()}
            hosp_options = list(config.HOSPITAL_NAMES.values())
            pref_hosp_name = config.HOSPITAL_NAMES.get(
                profile.get("pref_hospital", ""), hosp_options[0]
            ) if profile.get("pref_hospital","") in config.HOSPITAL_NAMES else hosp_options[0]
            selected_hosp_name = st.selectbox(
                "hospital", hosp_options,
                index=hosp_options.index(pref_hosp_name),
                label_visibility="collapsed",
                help="Choose the hospital you plan to visit",
            )
            selected_hosp_id = hosp_display[selected_hosp_name]

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            go_btn = st.button("Get My Estimate  →", type="primary", use_container_width=True)

        with col_result:
            # Compute + cache in session_state so results survive dialog/loading reruns.
            # Consent is asked at most once per session — later estimates reuse that choice.
            if go_btn:
                los_low, los_high   = predict_los(age, severity, admission_type, has_chronic, department)
                cost_low, cost_high = predict_cost(age, severity, admission_type, has_chronic, department)
                _occ_now = float(
                    occ_h[occ_h["hospital_id"] == selected_hosp_id]["occupancy_rate"].mean()
                    if "hospital_id" in occ_h.columns and selected_hosp_id in occ_h["hospital_id"].values
                    else 0.75
                )
                st.session_state["_est_pending"] = dict(
                    username=uname, age=age, gender=gender, severity=severity,
                    admission_type=admission_type, department=department,
                    hospital_id=selected_hosp_id, has_chronic=has_chronic,
                    has_insurance=has_insurance, visit_for=visit_for,
                    travel_dist=travel_dist, los_low=los_low, los_high=los_high,
                    cost_low=cost_low, cost_high=cost_high,
                    selected_hosp_name=selected_hosp_name,
                    district=profile.get("district", ""),
                    num_chronic=int(num_chronic),
                    hosp_occupancy_at_time=_occ_now,
                )
                if st.session_state.get("_consent_choice") is None:
                    st.session_state["_est_awaiting_consent"] = True
                else:
                    st.session_state["_est_awaiting_consent"] = False
                    st.session_state["_est_loading"] = True
                    st.session_state["_est_saved"] = (
                        save_estimate_request(st.session_state["_est_pending"])
                        if st.session_state["_consent_choice"] else False
                    )

            if st.session_state.get("_est_awaiting_consent"):
                _consent_popup(st.session_state["_est_pending"], t)

            if st.session_state.get("_est_loading"):
                with st.spinner("Calculating your personalised estimate…"):
                    _time.sleep(1.4)
                st.session_state["_est"] = st.session_state["_est_pending"]
                st.session_state["_est_loading"] = False
                st.rerun()

            if "_est" in st.session_state:
                _e = st.session_state["_est"]
                los_low   = _e["los_low"];   los_high  = _e["los_high"]
                cost_low  = _e["cost_low"];  cost_high = _e["cost_high"]
                los_mid   = (los_low + los_high) / 2
                selected_hosp_id   = _e["hospital_id"]
                selected_hosp_name = _e["selected_hosp_name"]
                severity    = _e["severity"]
                department  = _e["department"]
                has_chronic = _e["has_chronic"]
                visit_for   = _e["visit_for"]
                has_insurance = _e["has_insurance"]
                travel_dist = _e["travel_dist"]
                gender      = _e["gender"]
                district    = _e.get("district", "")
                num_chronic = _e.get("num_chronic", 0)
                hosp_occ_saved = _e.get("hosp_occupancy_at_time")

                st.markdown(result_hero(los_low, los_high, cost_low, cost_high,
                                        severity, department, t), unsafe_allow_html=True)

                # ── Department availability check ─────────────────────────
                hosp_depts   = dept_map.get(selected_hosp_id, [])
                dept_ok      = department in hosp_depts
                if not dept_ok:
                    alts = [config.HOSPITAL_NAMES.get(h, h)
                            for h in hospitals_with_department(department)
                            if h != selected_hosp_id]
                    alt_txt = (", ".join(alts[:3]) + " also offer it") if alts else "no listed alternative found"
                    st.warning(
                        f"⚠️ **{selected_hosp_name}** may not have a **{department}** department. "
                        f"{alt_txt}."
                    )

                # ── Recovery rate for this severity ────────────────────────
                rec_rate_sev = float(
                    (adm[adm["severity"] == severity]["discharge_outcome"] == "Recovered").mean()
                )

                # ── Best hour to arrive ────────────────────────────────────
                hourly = (adm[adm["hospital_id"] == selected_hosp_id]
                          .groupby("admission_hour")["length_of_stay_days"].count()
                          if selected_hosp_id in adm["hospital_id"].values
                          else adm.groupby("admission_hour")["length_of_stay_days"].count())
                quiet_hour = int(hourly.idxmin()) if len(hourly) else 9
                quiet_label = (f"{quiet_hour}:00–{quiet_hour+1}:00"
                               if quiet_hour < 23 else "23:00–00:00")

                # ── Hospital status card ──────────────────────────────────
                hosp_occ_val = float(
                    occ_h[occ_h["hospital_id"] == selected_hosp_id]["occupancy_rate"].mean()
                    if "hospital_id" in occ_h.columns and selected_hosp_id in occ_h["hospital_id"].values
                    else 0.75
                )
                hosp_avg_cost = float(
                    adm[adm["hospital_id"] == selected_hosp_id]["total_bill_npr"].mean()
                    if selected_hosp_id in adm["hospital_id"].values else (cost_low + cost_high) / 2
                )
                if hosp_occ_val > 0.85:
                    occ_color = "#EF4444"; occ_bg = "#FEE2E2"
                    occ_label = "Very busy right now"; occ_icon = "🔴"
                    occ_advice = "Expect longer waits. If your visit isn't urgent, consider another hospital or time."
                elif hosp_occ_val > 0.75:
                    occ_color = "#F59E0B"; occ_bg = "#FEF3C7"
                    occ_label = "Moderately busy"; occ_icon = "🟡"
                    occ_advice = "Reasonably busy. Book ahead if you can — don't just walk in."
                else:
                    occ_color = "#10B981"; occ_bg = "#D1FAE5"
                    occ_label = "Good availability"; occ_icon = "🟢"
                    occ_advice = "Good time to visit. Beds are available and waits are usually shorter."

                location    = config.HOSPITAL_LOCATIONS.get(selected_hosp_id, "Bagmati Region")
                hosp_level  = config.HOSPITAL_LEVELS.get(selected_hosp_id, "")
                hosp_type   = config.HOSPITAL_TYPES.get(selected_hosp_id, "")
                total_beds  = config.HOSPITAL_BEDS.get(selected_hosp_id, 0)
                beds_used   = int(hosp_occ_val * total_beds) if total_beds else 0
                beds_free   = max(0, total_beds - beds_used)
                level_badge = (
                    f'<span style="background:#EEF2FF;color:#4F46E5;font-size:.62rem;font-weight:700;'
                    f'padding:2px 8px;border-radius:9999px;margin-left:6px;">{hosp_level}</span>'
                    if hosp_level else ""
                )
                st.markdown(
                    f'<div style="background:#fff;border:1px solid {t["border"]};border-radius:14px;'
                    f'padding:16px 18px;margin-bottom:10px;">'
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
                    f'<div style="width:38px;height:38px;border-radius:10px;background:{t["p_light"]};'
                    f'display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;">🏥</div>'
                    f'<div style="flex:1;min-width:0;">'
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.88rem;'
                    f'font-weight:700;color:{t["t1"]};">{selected_hosp_name}{level_badge}</div>'
                    f'<div style="font-size:.72rem;color:{t["t3"]};">📍 {location} &nbsp;·&nbsp; {hosp_type} &nbsp;·&nbsp; {total_beds} beds total</div>'
                    f'</div>'
                    f'<div style="background:{occ_bg};border-radius:9999px;flex-shrink:0;'
                    f'padding:4px 12px;font-size:.72rem;font-weight:700;color:{occ_color};">'
                    f'{occ_icon} {occ_label}</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:10px;margin-bottom:12px;">'
                    f'<div style="flex:1;background:{t["p_light"]};border-radius:10px;padding:12px;text-align:center;">'
                    f'<div style="font-size:.68rem;color:{t["t3"]};margin-bottom:4px;">Beds available now</div>'
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.4rem;'
                    f'font-weight:800;color:{occ_color};">{beds_free}</div>'
                    f'<div style="font-size:.65rem;color:{t["t3"]};">out of {total_beds} total</div>'
                    f'</div>'
                    f'<div style="flex:1;background:#FFFBEB;border-radius:10px;padding:12px;text-align:center;">'
                    f'<div style="font-size:.68rem;color:{t["t3"]};margin-bottom:4px;">How busy it is</div>'
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.4rem;'
                    f'font-weight:800;color:#92400E;">{hosp_occ_val:.0%}</div>'
                    f'<div style="font-size:.65rem;color:{t["t3"]};">of beds occupied</div>'
                    f'</div>'
                    f'<div style="flex:1;background:#F0FDF4;border-radius:10px;padding:12px;text-align:center;">'
                    f'<div style="font-size:.68rem;color:{t["t3"]};margin-bottom:4px;">Avg cost here</div>'
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.1rem;'
                    f'font-weight:800;color:#065F46;">Rs. {hosp_avg_cost:,.0f}</div>'
                    f'<div style="font-size:.65rem;color:{t["t3"]};">per admission</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="font-size:.78rem;color:{t["t2"]};background:{occ_bg};'
                    f'border-radius:8px;padding:10px 12px;">'
                    f'{occ_advice}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ── Plain-language next-steps card ────────────────────────
                nights_low  = max(0, int(los_low))
                nights_high = max(0, int(los_high))
                if nights_high <= 3:
                    stay_tip = "This looks like a short stay. You may be able to arrange transport home within a few days."
                elif nights_high <= 10:
                    stay_tip = "Plan for about a week. Let your family or employer know you may be away for several days."
                else:
                    stay_tip = "This is a longer stay. Consider arranging someone to help at home after you're discharged."

                st.markdown(
                    f'<div style="background:#fff;border:1px solid {t["border"]};border-radius:14px;'
                    f'padding:16px 18px;margin-bottom:10px;">'
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.82rem;'
                    f'font-weight:700;color:{t["t1"]};margin-bottom:10px;">What this means for you</div>'
                    f'<div style="display:flex;flex-direction:column;gap:8px;">'
                    f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:.8rem;color:{t["t2"]};">'
                    f'<span style="font-size:1rem;flex-shrink:0;">🛏️</span>'
                    f'<span>{stay_tip}</span></div>'
                    f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:.8rem;color:{t["t2"]};">'
                    f'<span style="font-size:1rem;flex-shrink:0;">💰</span>'
                    f'<span>Budget between <strong>Rs. {cost_low:,.0f}</strong> and '
                    f'<strong>Rs. {cost_high:,.0f}</strong>. Ask the hospital about payment plans if needed.</span></div>'
                    f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:.8rem;color:{t["t2"]};">'
                    f'<span style="font-size:1rem;flex-shrink:0;">📋</span>'
                    f'<span>These are estimates only — your doctor will give you the exact plan. '
                    f'Bring this to your appointment as a starting point.</span></div>'
                    f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:.8rem;color:{t["t2"]};">'
                    f'<span style="font-size:1rem;flex-shrink:0;">✅</span>'
                    f'<span>For patients with <strong>{severity}</strong> conditions, about '
                    f'<strong>{rec_rate_sev:.0%}</strong> go home fully recovered — a good sign.</span></div>'
                    f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:.8rem;color:{t["t2"]};">'
                    f'<span style="font-size:1rem;flex-shrink:0;">🕐</span>'
                    f'<span>The quietest time to arrive at this hospital is around '
                    f'<strong>{quiet_label}</strong> — shorter queues, faster attention.</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

                # ── Personalised extra tips ───────────────────────────────
                extra_tips = []

                if gender == "Female":
                    extra_tips.append(("👩","For female patients",
                        "Make sure to ask the hospital about their maternity or women's health services when you call ahead. "
                        "Some departments have separate waiting areas for women."))
                elif gender == "Male":
                    extra_tips.append(("👨","For male patients",
                        "Men often delay seeking care — arriving at the right time matters more than you think. "
                        "If your symptoms have been going on for more than 2 days, do not wait longer."))

                if visit_for == "A child":
                    extra_tips.append(("👶","Bringing a child",
                        "A parent or legal guardian must be present at all times. Ask the hospital about their "
                        "<strong>Pediatrics ward</strong> when booking — children are seen separately from adults. "
                        "Bring the child's vaccination card and any previous medical notes."))
                elif visit_for == "A family member":
                    extra_tips.append(("👨‍👩‍👦","Planning for someone else",
                        "Bring their national ID or citizenship card, any existing prescriptions, "
                        "and a list of medications they currently take. You may be asked to sign consent forms on their behalf."))

                if has_insurance:
                    extra_tips.append(("🛡️","You have health insurance",
                        "Your insurer may cover part or all of your hospital costs. Before you go: "
                        "call your insurer to confirm the hospital is <strong>in-network</strong>, ask about your deductible, "
                        "and request a <em>prior authorisation</em> if your visit is planned. Keep all receipts."))
                else:
                    extra_tips.append(("💳","No insurance",
                        "Ask the hospital's billing desk about a <strong>payment plan</strong> — many hospitals in the "
                        "Bagmati region offer instalment options. Government hospitals (like district hospitals) "
                        "usually have lower fees than private ones."))

                if travel_dist == "More than 1 hour away":
                    extra_tips.append(("🚌","Travelling far",
                        "Since you are coming from a distance, call the hospital the day before to confirm "
                        "your appointment and check bed availability. Consider arranging a place for family "
                        "to stay nearby during your admission."))

                if extra_tips:
                    st.markdown(
                        f'<div style="background:#fff;border:1px solid {t["border"]};border-radius:14px;'
                        f'padding:16px 18px;margin-bottom:10px;">'
                        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.82rem;'
                        f'font-weight:700;color:{t["t1"]};margin-bottom:10px;">Just for you</div>'
                        f'<div style="display:flex;flex-direction:column;gap:10px;">',
                        unsafe_allow_html=True,
                    )
                    for ico, heading, body in extra_tips:
                        st.markdown(
                            f'<div style="display:flex;align-items:flex-start;gap:10px;">'
                            f'<div style="width:32px;height:32px;border-radius:8px;background:{t["p_light"]};'
                            f'display:flex;align-items:center;justify-content:center;'
                            f'font-size:1rem;flex-shrink:0;">{ico}</div>'
                            f'<div>'
                            f'<div style="font-size:.78rem;font-weight:700;color:{t["t1"]};margin-bottom:2px;">{heading}</div>'
                            f'<div style="font-size:.76rem;color:{t["t2"]};line-height:1.55;">{body}</div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div></div>', unsafe_allow_html=True)

                st.markdown(trust_note(t), unsafe_allow_html=True)

                # ── Consent & save confirmation (consent itself is resolved before
                # the estimate is ever shown — see the go_btn / _est_loading gate above) ──
                if st.session_state.get("_est_saved") is True:
                    st.markdown(
                        f'<div style="background:#F0FDF4;border:1px solid rgba(16,185,129,.2);'
                        f'border-left:4px solid #10B981;border-radius:10px;'
                        f'padding:10px 14px;font-size:.78rem;color:#065F46;margin-top:8px;">'
                        f'✅ <strong>Saved.</strong> Thank you — your estimate was saved anonymously '
                        f'to help improve the system for everyone.</div>',
                        unsafe_allow_html=True,
                    )
                elif st.session_state.get("_est_saved") is False:
                    st.markdown(
                        f'<div style="background:{t["p_light"]};border-radius:10px;'
                        f'padding:9px 13px;font-size:.76rem;color:{t["t3"]};margin-top:8px;">'
                        f'Not saved — no problem. Your estimate is still shown above.</div>',
                        unsafe_allow_html=True,
                    )

                # ── Booking button ────────────────────────────────────────────
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                if st.session_state.get("_booking_done"):
                    ref = st.session_state.get("_last_booking_ref", "")
                    st.markdown(
                        f'<div style="background:#F0FDF4;border:1px solid #10B981;'
                        f'border-left:4px solid #10B981;border-radius:12px;'
                        f'padding:14px 18px;margin-top:4px;">'
                        f'<div style="font-weight:700;color:#065F46;font-size:.9rem;margin-bottom:4px;">'
                        f'✅ Booking Request Submitted!</div>'
                        f'<div style="font-size:.8rem;color:#065F46;">Reference: '
                        f'<strong>{ref}</strong> · The hospital admin will confirm your slot. '
                        f'Check <strong>My Bookings</strong> tab for updates.</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button("📅  Book This Visit", use_container_width=True,
                                 key="open_booking_dialog"):
                        st.session_state["_booking_done"] = False
                        _booking_dialog(st.session_state["_est"], profile, t)

            else:
                st.markdown(f"""
                <div style="background:{t['p_light']};border:2px dashed {t['border']};
                  border-radius:20px;padding:52px 24px;text-align:center;">
                  <div style="font-size:3rem;margin-bottom:16px;">🏥</div>
                  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;
                    font-size:1.05rem;color:{t['t1']};margin-bottom:8px;">
                    Your personal estimate will appear here
                  </div>
                  <div style="font-size:.85rem;color:{t['t2']};line-height:1.6;max-width:300px;margin:0 auto;">
                    Fill in your details on the left, then tap
                    <strong style="color:{t['primary']};">Get My Estimate</strong>.
                    It only takes 30 seconds.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ════ TAB 2: WHAT TO EXPECT ════════════════════════════════════════════
    with tab2:
        row1_a, row1_b = st.columns(2, gap="medium")

        with row1_a:
            st.markdown(f"""<div class="section-card">
              {section_hdr(t['p_light'],'🛏️','How long do patients usually stay?',
                           'Typical nights in hospital — half of patients leave sooner than this',t)}
            """, unsafe_allow_html=True)
            sev_los = (adm.groupby("severity")["length_of_stay_days"]
                       .median().reset_index()
                       .rename(columns={"length_of_stay_days": "typical_los"}))
            order = ["Mild","Moderate","Severe","Critical"]
            sev_los["severity"] = pd.Categorical(sev_los["severity"], categories=order, ordered=True)
            sev_los = sev_los.sort_values("severity")
            sev_colors = ["#10B981","#F59E0B","#F97316","#EF4444"]
            fig = go.Figure(go.Bar(
                x=sev_los["severity"], y=sev_los["typical_los"],
                marker=dict(color=sev_colors, cornerradius=10,
                            line=dict(color="rgba(0,0,0,0)", width=0)),
                text=sev_los["typical_los"],
                texttemplate="<b>%{text:.0f} nights</b>", textposition="outside",
                textfont=dict(size=12, color=t["t2"]),
                hovertemplate="<b>%{x}</b><br>Typical stay: %{y:.0f} nights<br><i>Half of patients leave sooner</i><extra></extra>",
            ))
            fig.update_layout(**chart_layout(t, 300))
            pchart(fig)
            mild_med  = int(sev_los[sev_los["severity"]=="Mild"]["typical_los"].iloc[0]) if "Mild" in sev_los["severity"].values else 4
            crit_med  = int(sev_los[sev_los["severity"]=="Critical"]["typical_los"].iloc[0]) if "Critical" in sev_los["severity"].values else 17
            st.markdown(
                f'<p style="font-size:.78rem;color:{t["t3"]};margin:4px 0 0;padding:0 2px;">'
                f'💡 Half of patients with a <strong>Mild</strong> condition leave within <strong>{mild_med} nights</strong>. '
                f'Critical cases typically need around <strong>{crit_med} nights</strong>.'
                f'</p></div>',
                unsafe_allow_html=True
            )

        with row1_b:
            st.markdown(f"""<div class="section-card">
              {section_hdr(t['p_light'],'🚪','How do most people come in?',
                           'Based on 120,000+ admissions',t)}
            """, unsafe_allow_html=True)
            at = adm["admission_type"].value_counts().reset_index()
            at.columns = ["type", "count"]
            fig2 = go.Figure(go.Pie(
                labels=at["type"], values=at["count"],
                hole=0.60,
                marker=dict(colors=["#6366F1","#10B981","#F59E0B","#EC4899"],
                            line=dict(color="#fff", width=3)),
                textinfo="label+percent", textfont=dict(size=11),
                pull=[0.03, 0, 0, 0],
                hovertemplate="<b>%{label}</b><br>%{value:,} admissions<br>%{percent}<extra></extra>",
            ))
            fig2.update_layout(**chart_layout(t, 300))
            pchart(fig2)
            top_type = at.iloc[0]["type"] if len(at) else "Emergency"
            st.markdown(f"""<p style="font-size:.78rem;color:{t['t3']};margin:4px 0 0;padding:0 2px;">
              💡 Most patients arrive via <strong>{top_type}</strong>. If your visit is planned, you'll
              likely have a shorter wait than emergency patients.
            </p></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="section-card" style="margin-top:14px;">
          {section_hdr(t['p_light'],'🩺','What are the most common reasons people visit?',
                       'Top 10 health conditions — each bar = number of admissions',t)}
        """, unsafe_allow_html=True)
        diag = adm["diagnosis_category"].value_counts().head(10).reset_index()
        diag.columns = ["diagnosis", "count"]
        colors_d = ["#6366F1","#8B5CF6","#A78BFA","#EC4899","#F43F5E",
                    "#10B981","#059669","#F59E0B","#0EA5E9","#06B6D4"]
        fig3 = go.Figure(go.Bar(
            y=diag["diagnosis"], x=diag["count"], orientation="h",
            marker=dict(color=colors_d[:len(diag)], cornerradius=6,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            text=diag["count"], texttemplate="<b>%{text:,} visits</b>", textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>%{x:,} admissions<extra></extra>",
        ))
        fig3.update_layout(**chart_layout(t, 360))
        pchart(fig3)
        top_diag = diag.iloc[0]["diagnosis"] if len(diag) else "Cardiovascular"
        st.markdown(f"""<p style="font-size:.78rem;color:{t['t3']};margin:6px 0 0;padding:0 2px;">
          💡 <strong>{top_diag}</strong> is the most common reason for hospital visits in this region.
          If that matches your condition, the estimates in the Planner tab will be especially relevant.
        </p></div>""", unsafe_allow_html=True)

    # ════ TAB 3: BEST TIME TO VISIT ════════════════════════════════════════
    with tab3:
        mo_avg = occ.groupby("month")["occupancy_rate"].mean()

        # ── Build month badges (no indentation — avoids Markdown code-block treatment) ──
        month_names_full = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
        month_icons = ["❄️","❄️","🌸","🌸","☀️","☀️","🌧️","🌧️","🍂","🍂","❄️","❄️"]
        badges_html = ""
        for i, mname in enumerate(month_names_full):
            m = i + 1
            occ_val = float(mo_avg.get(m, 0.75))
            if occ_val > 0.85:
                bg_c = "#FEE2E2"; txt_c = "#991B1B"; brd_c = "#EF4444"; lbl = "Very busy"
            elif occ_val > 0.75:
                bg_c = "#FEF3C7"; txt_c = "#92400E"; brd_c = "#F59E0B"; lbl = "Busy"
            else:
                bg_c = "#D1FAE5"; txt_c = "#065F46"; brd_c = "#10B981"; lbl = "Good time"
            pct_str = f"{occ_val:.0%}"
            ico = month_icons[i]
            # single line — no leading spaces → Markdown won't treat as code
            badges_html += (
                f'<div style="flex:1;min-width:60px;background:{bg_c};border-radius:12px;'
                f'padding:12px 6px;text-align:center;border:1px solid {brd_c}44;">'
                f'<div style="font-size:1.1rem;">{ico}</div>'
                f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.75rem;'
                f'font-weight:700;color:{txt_c};margin:4px 0 2px;">{mname}</div>'
                f'<div style="font-size:.72rem;font-weight:700;color:{txt_c};">{pct_str}</div>'
                f'<div style="font-size:.6rem;color:{txt_c};opacity:.75;margin-top:2px;">{lbl}</div>'
                f'</div>'
            )

        legend_html = (
            f'<div style="display:flex;align-items:center;gap:14px;font-size:.74rem;'
            f'color:{t["t3"]};padding:10px 2px 0;">'
            f'<span>&#11044; <span style="color:#10B981;">&#9632;</span> Under 75% — quietest</span>'
            f'<span>&#11044; <span style="color:#F59E0B;">&#9632;</span> 75–85% — book ahead</span>'
            f'<span>&#11044; <span style="color:#EF4444;">&#9632;</span> Over 85% — avoid if possible</span>'
            f'</div>'
        )

        hdr_html = section_hdr(
            t["p_light"], "📅",
            "Which month is best for a planned visit?",
            "Green = quiet &nbsp;·&nbsp; Yellow = busy &nbsp;·&nbsp; Red = very busy", t
        )
        st.markdown(
            f'<div class="section-card">{hdr_html}'
            f'<div style="display:flex;flex-wrap:nowrap;gap:8px;overflow-x:auto;">'
            f'{badges_html}'
            f'</div>'
            f'{legend_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Season tip cards (single-line HTML per card to avoid Markdown code-block) ──
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        seasons_data = [
            ("🌸","Spring","Mar–May","72%","#FEF9C3","#92400E","#F59E0B",
             "A good window. Book 2–3 weeks ahead for planned visits."),
            ("☀️","Summer","Jun–Aug","68%","#D1FAE5","#065F46","#10B981",
             "Best time of year. Shortest waits, most bed availability."),
            ("🌧️","Monsoon","Jul–Sep","84%","#FEE2E2","#991B1B","#EF4444",
             "Very busy. Illness spikes — avoid non-urgent visits if you can."),
            ("🍂","Autumn","Sep–Nov","79%","#FEF3C7","#92400E","#F59E0B",
             "Busy period. Go for urgent needs only; waits may be longer."),
            ("❄️","Winter","Dec–Feb","75%","#EFF6FF","#1E40AF","#6366F1",
             "Moderate. Cold-related illnesses push up demand — plan ahead."),
        ]
        cols = st.columns(5, gap="small")
        for col, (ico, season, months, pct, bg, txt, brd, body) in zip(cols, seasons_data):
            col.markdown(
                f'<div style="background:{bg};border-top:3px solid {brd};border:1px solid {brd}44;'
                f'border-radius:14px;padding:14px 12px;height:100%;">'
                f'<div style="font-size:1.5rem;margin-bottom:6px;">{ico}</div>'
                f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:.82rem;'
                f'font-weight:700;color:{txt};">{season}</div>'
                f'<div style="font-size:.7rem;color:{txt};opacity:.7;margin-bottom:6px;">{months}</div>'
                f'<div style="font-size:1.2rem;font-weight:800;color:{txt};'
                f'font-family:\'Plus Jakarta Sans\',sans-serif;">{pct} full</div>'
                f'<div style="font-size:.73rem;color:{txt};opacity:.8;margin-top:8px;'
                f'line-height:1.5;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ════ TAB 4: PLAN BY AGE ═══════════════════════════════════════════════
    with tab4:
        merged   = merged_admissions_patients()
        age_order = ["Infant", "Toddler", "Child", "Adolescent",
                     "Young Adult", "Middle Age", "Senior", "Elderly"]
        elderly_groups = {"Senior", "Elderly"}

        st.markdown(f"""
        <div style="background:{t['p_light']};border:1px solid {t['border']};border-radius:14px;
          padding:14px 20px;font-size:.82rem;color:{t['t2']};margin-bottom:18px;
          display:flex;align-items:flex-start;gap:10px;">
          <span style="font-size:1.1rem;">👋</span>
          <div>Planning care for an older family member? <strong>Senior and Elderly</strong> patients
          typically stay longer and pay more — useful to know when budgeting.</div>
        </div>""", unsafe_allow_html=True)

        ag_a, ag_b = st.columns(2, gap="medium")

        # ── shared data ──────────────────────────────────────────────────────
        al = (merged.groupby("age_group")["length_of_stay_days"].median().reset_index()
              .rename(columns={"length_of_stay_days": "typical_los"}))
        al["age_group"] = pd.Categorical(al["age_group"],
            categories=[a for a in age_order if a in al["age_group"].values], ordered=True)
        al = al.sort_values("age_group").reset_index(drop=True)

        ac2 = (merged.groupby("age_group")["total_bill_npr"].mean().reset_index()
               .rename(columns={"total_bill_npr": "avg_cost"}))
        ac2["age_group"] = pd.Categorical(ac2["age_group"],
            categories=[a for a in age_order if a in ac2["age_group"].values], ordered=True)
        ac2 = ac2.sort_values("age_group").reset_index(drop=True)

        age_colors_los  = ["#EF4444" if g in elderly_groups else "#6366F1" for g in al["age_group"]]
        age_colors_cost = ["#EF4444" if g in elderly_groups else "#10B981" for g in ac2["age_group"]]

        with ag_a:
            st.markdown(
                f'<div class="section-card">'
                f'{section_hdr(t["p_light"],"🛏️","How long do different age groups usually stay?","Lollipop — dot = typical nights · red = Senior / Elderly",t)}',
                unsafe_allow_html=True,
            )
            # Lollipop: horizontal stems + coloured dots
            fig_lo = go.Figure()
            for i, row in al.iterrows():
                fig_lo.add_shape(
                    type="line", layer="below",
                    x0=0, x1=float(row["typical_los"]),
                    y0=i, y1=i,
                    line=dict(color=age_colors_los[i], width=2, dash="dot"),
                )
            fig_lo.add_trace(go.Scatter(
                x=al["typical_los"],
                y=list(range(len(al))),
                mode="markers+text",
                marker=dict(color=age_colors_los, size=18,
                            line=dict(color="#fff", width=2)),
                text=[f"<b>{int(v)}d</b>" for v in al["typical_los"]],
                textposition="middle right",
                textfont=dict(size=11, color=t["t1"]),
                customdata=al["age_group"].astype(str),
                hovertemplate="<b>%{customdata}</b><br>Typical stay: %{x:.0f} nights<extra></extra>",
            ))
            lo_layout = chart_layout(t, 310)
            lo_layout["yaxis"] = dict(
                tickvals=list(range(len(al))),
                ticktext=al["age_group"].astype(str).tolist(),
                showgrid=False, zeroline=False,
                linecolor="rgba(0,0,0,0)",
                tickfont=dict(size=11, color=t["t3"]),
            )
            lo_layout["xaxis"] = dict(
                showgrid=True, gridcolor="rgba(0,0,0,.05)",
                zeroline=True, zerolinecolor="rgba(0,0,0,.1)",
                linecolor="rgba(0,0,0,0)",
                tickfont=dict(size=10, color=t["t3"]),
                ticksuffix=" nights",
            )
            lo_layout["showlegend"] = False
            fig_lo.update_layout(**lo_layout)
            pchart(fig_lo, key="age_los_lollipop")
            st.markdown(
                f'<p style="font-size:.78rem;color:{t["t3"]};margin:4px 0 0;padding:0 2px;">'
                f'💡 Young adults typically leave in {int(al[al["age_group"]=="Young Adult"]["typical_los"].iloc[0]) if "Young Adult" in al["age_group"].values else "~5"} nights. '
                f'Senior & Elderly patients often need twice as long.</p></div>',
                unsafe_allow_html=True,
            )

        with ag_b:
            st.markdown(
                f'<div class="section-card">'
                f'{section_hdr("#FFFBEB","💰","How do costs rise with age?","Bubble size = number of patients · colour = cost level (green → red)",t)}',
                unsafe_allow_html=True,
            )
            # Enrich ac2 with patient counts
            _cnt = (merged.groupby("age_group")["patient_id"]
                    .count().reset_index().rename(columns={"patient_id": "count"}))
            ac2  = ac2.merge(_cnt, on="age_group", how="left")
            ac2["count"] = ac2["count"].fillna(1)

            # Bubble sizes: scaled so smallest ≈ 20px, largest ≈ 55px diameter
            _min_c, _max_c = ac2["count"].min(), ac2["count"].max()
            bubble_sz = [20 + 35 * ((c - _min_c) / max(_max_c - _min_c, 1))
                         for c in ac2["count"]]

            fig_co = go.Figure(go.Scatter(
                x=ac2["age_group"].astype(str),
                y=ac2["avg_cost"],
                mode="markers+text",
                marker=dict(
                    size=bubble_sz,
                    sizemode="diameter",
                    color=ac2["avg_cost"],
                    colorscale=[
                        [0.0, "#10B981"],
                        [0.5, "#F59E0B"],
                        [1.0, "#EF4444"],
                    ],
                    cmin=float(ac2["avg_cost"].min()),
                    cmax=float(ac2["avg_cost"].max()),
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Avg cost", font=dict(size=10, color=t["t3"])),
                        thickness=10, len=0.7,
                        tickprefix="Rs.",
                        tickformat=",",
                        tickfont=dict(size=9, color=t["t3"]),
                        outlinewidth=0,
                    ),
                    line=dict(color="#fff", width=2),
                ),
                text=["<b>Rs. " + f"{int(v):,}</b>" for v in ac2["avg_cost"]],
                textposition="top center",
                textfont=dict(size=9, color=t["t1"]),
                customdata=list(zip(
                    ac2["age_group"].astype(str),
                    ac2["count"].astype(int),
                    ac2["avg_cost"],
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Avg cost: Rs. %{customdata[2]:,.0f}<br>"
                    "%{customdata[1]:,} patients<extra></extra>"
                ),
            ))
            co_layout = chart_layout(t, 320)
            co_layout["yaxis"] = dict(
                showgrid=True, gridcolor="rgba(0,0,0,.04)",
                zeroline=False, linecolor="rgba(0,0,0,0)",
                tickfont=dict(size=9, color=t["t3"]),
                tickprefix="Rs. ", tickformat=",",
                range=[
                    float(ac2["avg_cost"].min()) * 0.97,
                    float(ac2["avg_cost"].max()) * 1.07,
                ],
            )
            co_layout["xaxis"] = dict(
                showgrid=False, zeroline=False,
                linecolor="rgba(0,0,0,0)",
                tickfont=dict(size=10, color=t["t3"]),
                tickangle=-15,
            )
            co_layout["margin"] = dict(l=70, r=60, t=14, b=50)
            co_layout["showlegend"] = False
            fig_co.update_layout(**co_layout)
            pchart(fig_co, key="age_cost_bubble")
            st.markdown(
                f'<p style="font-size:.78rem;color:{t["t3"]};margin:4px 0 0;padding:0 2px;">'
                f'💡 Bigger bubble = more patients in that group. Colour shifts from green (lower cost) to red (higher cost).'
                f'</p></div>',
                unsafe_allow_html=True,
            )

        # bottom insight
        elderly_cost = float(ac2[ac2["age_group"].isin(["Elderly","Senior"])]["avg_cost"].mean()) if any(g in ac2["age_group"].values for g in ["Elderly","Senior"]) else 0
        young_cost   = float(ac2[ac2["age_group"].isin(["Young Adult","Adolescent"])]["avg_cost"].mean()) if any(g in ac2["age_group"].values for g in ["Young Adult","Adolescent"]) else 1
        cost_mult    = elderly_cost / young_cost if young_cost else 1
        st.markdown(insight_box(
            "Planning care for an older family member?",
            f"Senior and Elderly patients typically pay around <strong>{cost_mult:.1f}× more</strong> "
            f"than young adults, and stay longer in hospital. "
            "Budget generously and try to visit during <strong>Summer (Jun–Aug)</strong> when hospitals are least crowded.",
            t
        ), unsafe_allow_html=True)

    # ════ TAB 5: COMPARE HOSPITALS ══════════════════════════════════════════
    with tab5:
        st.markdown(
            f'<div style="font-size:.82rem;color:{t["t2"]};margin-bottom:18px;">'
            f'Tap a hospital to see how busy it is, what it costs, and when to visit.</div>',
            unsafe_allow_html=True,
        )

        def _fv(v, default=0.0):
            return float(v) if pd.notna(v) else default

        h_ids = (list(hosp_df["hospital_id"].values)
                 if len(hosp_df) > 0 else list(config.HOSPITAL_NAMES.keys()))

        for row_start in range(0, len(h_ids), 3):
            group = h_ids[row_start : row_start + 3]
            cols  = st.columns(len(group), gap="medium")
            for col, hid in zip(cols, group):
                hname    = config.HOSPITAL_NAMES.get(hid, hid)
                hloc     = config.HOSPITAL_LOCATIONS.get(hid, "")
                hlvl     = config.HOSPITAL_LEVELS.get(hid, "")
                hbeds    = config.HOSPITAL_BEDS.get(hid, 0)

                hrow = (hosp_df[hosp_df["hospital_id"] == hid].iloc[0]
                        if hid in hosp_df["hospital_id"].values else {})

                occ_v     = _fv(occ_h[occ_h["hospital_id"] == hid]["occupancy_rate"].mean()
                                if "hospital_id" in occ_h.columns and hid in occ_h["hospital_id"].values
                                else 0.75, 0.75)
                beds_free = max(0, hbeds - int(occ_v * hbeds))
                rec_pct   = _fv(hrow.get("recovery_rate")  if hasattr(hrow, "get") else None)
                avg_cost  = _fv(hrow.get("avg_cost")       if hasattr(hrow, "get") else None)
                wait_min  = _fv(hrow.get("avg_wait_triage") if hasattr(hrow, "get") else None)
                rec_in_10 = round(rec_pct * 10)

                if occ_v > 0.85:
                    accent="#EF4444"; occ_lbl="Very busy";   occ_bg="#FEF2F2"
                elif occ_v > 0.75:
                    accent="#F59E0B"; occ_lbl="Moderately busy"; occ_bg="#FFFBEB"
                else:
                    accent="#10B981"; occ_lbl="Available";   occ_bg="#F0FDF4"

                lvl_color = "#4F46E5" if hlvl == "Tertiary" else "#7C3AED"
                occ_pct_w = int(occ_v * 100)

                with col:
                    st.markdown(
                        f'<div style="background:#fff;border:1px solid {t["border"]};'
                        f'border-radius:16px;overflow:hidden;margin-bottom:4px;'
                        f'box-shadow:0 2px 12px rgba(79,70,229,.08);">'

                        f'<div style="padding:16px 16px 0;">'

                        f'<div style="display:flex;align-items:flex-start;'
                        f'justify-content:space-between;gap:6px;margin-bottom:4px;">'
                        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;'
                        f'font-size:.92rem;font-weight:800;color:{t["t1"]};line-height:1.3;">'
                        f'{hname}</div>'
                        f'<span style="background:#EEF2FF;color:{lvl_color};font-size:.56rem;'
                        f'font-weight:700;padding:3px 8px;border-radius:9999px;'
                        f'white-space:nowrap;flex-shrink:0;margin-top:2px;">{hlvl}</span>'
                        f'</div>'

                        f'<div style="font-size:.68rem;color:{t["t3"]};margin-bottom:12px;">'
                        f'📍 {hloc}</div>'

                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:12px;">'

                        f'<div style="background:{t["p_light"]};border-radius:9px;padding:8px 10px;">'
                        f'<div style="font-size:.6rem;color:{t["t3"]};margin-bottom:1px;">🛏️ Beds free</div>'
                        f'<div style="font-size:.88rem;font-weight:800;color:{t["t1"]};">{beds_free}'
                        f'<span style="font-size:.6rem;font-weight:500;color:{t["t3"]};"> / {hbeds}</span></div>'
                        f'</div>'

                        f'<div style="background:{t["p_light"]};border-radius:9px;padding:8px 10px;">'
                        f'<div style="font-size:.6rem;color:{t["t3"]};margin-bottom:1px;">❤️ Go home well</div>'
                        f'<div style="font-size:.88rem;font-weight:800;color:#10B981;">{rec_in_10}'
                        f'<span style="font-size:.6rem;font-weight:500;color:{t["t3"]};"> out of 10</span></div>'
                        f'</div>'

                        f'<div style="background:{t["p_light"]};border-radius:9px;padding:8px 10px;">'
                        f'<div style="font-size:.6rem;color:{t["t3"]};margin-bottom:1px;">💰 Typical cost</div>'
                        f'<div style="font-size:.88rem;font-weight:800;color:{t["t1"]};">Rs. {avg_cost:,.0f}</div>'
                        f'</div>'

                        f'<div style="background:{t["p_light"]};border-radius:9px;padding:8px 10px;">'
                        f'<div style="font-size:.6rem;color:{t["t3"]};margin-bottom:1px;">⏱️ Wait time</div>'
                        f'<div style="font-size:.88rem;font-weight:800;color:{t["t1"]};">~{wait_min:.0f}'
                        f'<span style="font-size:.6rem;font-weight:500;color:{t["t3"]};"> min</span></div>'
                        f'</div>'

                        f'</div>'

                        f'<div style="margin-bottom:12px;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:.62rem;color:{t["t3"]};margin-bottom:4px;">'
                        f'<span>How busy right now</span>'
                        f'<span style="font-weight:700;color:{accent};">{occ_lbl}</span></div>'
                        f'<div style="background:#F1F5F9;border-radius:9999px;height:6px;overflow:hidden;">'
                        f'<div style="width:{occ_pct_w}%;background:{accent};height:100%;'
                        f'border-radius:9999px;transition:width .3s;"></div>'
                        f'</div></div>'

                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("View Details →", key=f"hosp_btn_{hid}",
                                 use_container_width=True):
                        _hospital_popup(hid, occ_h, adm, hosp_df, dept_map, t)

    # ════ TAB 8: MY PROFILE ═════════════════════════════════════════════════
    with tab8:
        # uname / profile already loaded at top of patient_dashboard()
        joined    = get_user_created_at(uname)
        full_name = p_full

        # Flash message
        if st.session_state.get("profile_msg"):
            kind, pmsg = st.session_state.profile_msg
            if kind == "ok":
                st.success(pmsg)
            else:
                st.error(pmsg)
            st.session_state.profile_msg = None

        # ── Profile header card (navy, thumbnail avatar ~5 KB base64) ──────────
        def _esc(s):
            return str(s).replace("<", "&lt;").replace(">", "&gt;")
        full_name     = _esc(full_name)
        district_chip = _esc(profile.get("district") or "Bagmati Region")
        blood_chip    = _esc(profile.get("blood_type") or "Unknown")
        occ_chip      = _esc(profile.get("occupation") or "")
        occ_span      = (
            f'<span style="background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);'
            f'border-radius:9999px;padding:2px 11px;font-size:.7rem;color:rgba(255,255,255,.8);">'
            f'&#128188; {occ_chip}</span>'
        ) if occ_chip else ""
        av_html    = _avatar_html(uname, 84, t, full_name)
        patient_id = get_patient_id(uname)
        pid_chip   = (
            f'<span style="background:rgba(99,102,241,.25);border:1px solid rgba(99,102,241,.4);'
            f'border-radius:9999px;padding:2px 11px;font-size:.7rem;color:rgba(255,255,255,.9);'
            f'font-weight:700;letter-spacing:.03em;">&#127973; {patient_id}</span>'
        ) if patient_id else ""

        if uname.startswith("google_"):
            sso_email = uname[len("google_"):]
            sub_line = (
                f'<span style="background:rgba(255,255,255,.15);border-radius:9999px;'
                f'padding:1px 8px;font-size:.68rem;margin-right:6px;">G Google</span>'
                f'{sso_email} &#183; Member since {joined}'
            )
        else:
            sub_line = f'@{uname} &#183; Member since {joined}'

        st.markdown(
            f'<div style="background:linear-gradient(130deg,#0F2447 0%,#1B3A6B 100%);'
            f'border-radius:18px;padding:22px 26px;margin-bottom:18px;'
            f'display:flex;align-items:center;gap:20px;overflow:hidden;">'
            f'{av_html}'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:1.3rem;font-weight:800;'
            f'color:#fff;letter-spacing:-.02em;margin-bottom:2px;">{full_name}</div>'
            f'<div style="font-size:.8rem;color:rgba(255,255,255,.5);margin-bottom:10px;">{sub_line}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:7px;">'
            f'{pid_chip}'
            f'<span style="background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);'
            f'border-radius:9999px;padding:2px 11px;font-size:.7rem;color:rgba(255,255,255,.8);">'
            f'&#128205; {district_chip}</span>'
            f'<span style="background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);'
            f'border-radius:9999px;padding:2px 11px;font-size:.7rem;color:rgba(255,255,255,.8);">'
            f'&#129656; Blood: {blood_chip}</span>'
            f'{occ_span}'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Photo upload ─────────────────────────────────────────────────────
        with st.expander("&#128247;  Change profile photo", expanded=False):
            uploaded_photo = st.file_uploader(
                "Upload JPG or PNG (max 2 MB)", type=["jpg","jpeg","png"],
                key="profile_tab_avatar",
            )
            if uploaded_photo:
                if uploaded_photo.size > 2_000_000:
                    st.error("File too large — please keep it under 2 MB.")
                else:
                    img_bytes = uploaded_photo.read()
                    ext = uploaded_photo.name.rsplit(".", 1)[-1].lower()
                    ok, result = save_avatar(uname, img_bytes, ext)
                    if ok:
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {result}")

        # ── Inner tabs ───────────────────────────────────────────────────────
        ptab1, ptab2, ptab3 = st.tabs([
            "&#128100;  Personal Info",
            "&#129657;  Health Profile",
            "&#128274;  Security",
        ])

        # ── Personal Info ────────────────────────────────────────────────────
        with ptab1:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            with st.form("prof_personal_tab"):
                pf1, pf2 = st.columns(2)
                with pf1:
                    v_full = st.text_input("Full Name", value=profile.get("full_name",""))
                with pf2:
                    g_opts = ["Prefer not to say","Male","Female","Non-binary"]
                    saved_g = profile.get("gender","Prefer not to say") or "Prefer not to say"
                    v_gender = st.selectbox("Gender", g_opts,
                        index=g_opts.index(saved_g) if saved_g in g_opts else 0)

                pf3, pf4 = st.columns(2)
                with pf3:
                    v_dob = st.text_input("Date of Birth (YYYY-MM-DD)",
                        value=profile.get("dob",""), placeholder="e.g. 1995-08-22")
                with pf4:
                    d_opts = ["— Select —","Kathmandu","Lalitpur","Bhaktapur","Kavrepalanchok",
                              "Sindhupalchok","Nuwakot","Rasuwa","Dhading","Makwanpur",
                              "Chitwan","Sindhuli","Ramechhap","Dolakha","Other"]
                    saved_d = profile.get("district","— Select —") or "— Select —"
                    v_district = st.selectbox("District / City", d_opts,
                        index=d_opts.index(saved_d) if saved_d in d_opts else 0)

                pf5, pf6 = st.columns(2)
                with pf5:
                    v_phone = st.text_input("Phone", value=profile.get("phone",""),
                                            placeholder="+977 98XXXXXXXX")
                with pf6:
                    v_email = st.text_input("Email", value=profile.get("email",""),
                                            placeholder="you@example.com")

                h_opts  = ["Any hospital","Hospital 1 — Kathmandu","Hospital 2 — Lalitpur",
                           "Hospital 3 — Bhaktapur","Hospital 4 — Kavrepalanchok",
                           "Hospital 5 — Sindhupalchok"]
                saved_h = profile.get("pref_hospital","Any hospital") or "Any hospital"
                v_hosp  = st.selectbox("Preferred Hospital", h_opts,
                    index=h_opts.index(saved_h) if saved_h in h_opts else 0)

                sp_btn = st.form_submit_button(
                    "Save Personal Info", use_container_width=True, type="primary")

            if sp_btn:
                ok, msg = save_profile(uname, {
                    "full_name": v_full, "gender": v_gender, "dob": v_dob,
                    "district": v_district if v_district != "— Select —" else "",
                    "phone": v_phone, "email": v_email, "pref_hospital": v_hosp,
                })
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        # ── Health Profile ───────────────────────────────────────────────────
        with ptab2:
            st.markdown(f"""
            <div style="background:#FFFBEB;border:1px solid rgba(217,119,6,.18);border-radius:10px;
              padding:11px 15px;font-size:.8rem;color:#92400E;margin-bottom:14px;
              display:flex;align-items:flex-start;gap:9px;">
              <span>&#9888;&#65039;</span>
              <span>Used only to improve the AI planner estimates.
              <strong>Never shared or used for clinical decisions.</strong></span>
            </div>
            """, unsafe_allow_html=True)

            with st.form("prof_health_tab"):
                hf1, hf2 = st.columns(2)
                with hf1:
                    bt_opts  = ["Unknown","A+","A-","B+","B-","O+","O-","AB+","AB-"]
                    saved_bt = profile.get("blood_type","Unknown") or "Unknown"
                    v_blood  = st.selectbox("Blood Type", bt_opts,
                        index=bt_opts.index(saved_bt) if saved_bt in bt_opts else 0)
                with hf2:
                    o_opts  = ["— Select —","Student","Healthcare Worker","Government Employee",
                               "Private Sector","Farmer / Agriculture","Business Owner",
                               "Retired","Unemployed","Other"]
                    saved_o = profile.get("occupation","— Select —") or "— Select —"
                    v_occ   = st.selectbox("Occupation", o_opts,
                        index=o_opts.index(saved_o) if saved_o in o_opts else 0)

                v_allergies = st.text_area("Known Allergies",
                    value=profile.get("allergies",""),
                    placeholder="e.g. Penicillin, latex, peanuts — leave blank if none",
                    height=76)
                v_chronic = st.text_area("Chronic Conditions",
                    value=profile.get("chronic_conditions",""),
                    placeholder="e.g. Diabetes Type 2, Hypertension — leave blank if none",
                    height=76)

                st.markdown(f"""
                <div class="schip" style="margin-top:4px;">
                  <span class="schip-label">Emergency Contact</span>
                  <div class="schip-line"></div>
                </div>""", unsafe_allow_html=True)

                ec1, ec2 = st.columns(2)
                with ec1:
                    v_ec_name = st.text_input("Contact Name",
                        value=profile.get("ec_name",""), placeholder="Ramesh Thapa")
                with ec2:
                    r_opts  = ["—","Spouse / Partner","Parent","Sibling","Child","Guardian","Friend","Other"]
                    saved_r = profile.get("ec_relationship","—") or "—"
                    v_ec_rel = st.selectbox("Relationship", r_opts,
                        index=r_opts.index(saved_r) if saved_r in r_opts else 0)
                v_ec_phone = st.text_input("Contact Phone",
                    value=profile.get("ec_phone",""), placeholder="+977 98XXXXXXXX")

                sh_btn = st.form_submit_button(
                    "Save Health Profile", use_container_width=True, type="primary")

            if sh_btn:
                ok, msg = save_profile(uname, {
                    "blood_type": v_blood,
                    "occupation": v_occ if v_occ != "— Select —" else "",
                    "allergies": v_allergies,
                    "chronic_conditions": v_chronic,
                    "ec_name": v_ec_name,
                    "ec_relationship": v_ec_rel if v_ec_rel != "—" else "",
                    "ec_phone": v_ec_phone,
                })
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        # ── Security ─────────────────────────────────────────────────────────
        with ptab3:
            st.markdown(f"""
            <div style="background:{t['p_light']};border:1px solid {t['border']};border-radius:14px;
              padding:16px 18px;margin-bottom:16px;display:flex;align-items:flex-start;gap:13px;">
              <div style="width:38px;height:38px;border-radius:10px;background:{t['primary']};
                display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">
                &#128274;
              </div>
              <div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;
                  color:{t['t1']};margin-bottom:3px;">Change Password</div>
                <div style="font-size:.8rem;color:{t['t2']};line-height:1.6;">
                  Passwords are stored as <strong>SHA-256 salted hashes</strong> — never plain text.
                  Use at least 8 characters mixing letters, numbers and symbols.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            sec_col, _ = st.columns([1.1, 1])
            with sec_col:
                v_old  = st.text_input("Current Password", type="password",
                                       placeholder="Your current password",
                                       key="sec_tab_old")
                v_new1 = st.text_input("New Password", type="password",
                                       placeholder="e.g. Bagmati@2024",
                                       key="sec_tab_new1")
                v_new2 = st.text_input("Confirm New Password", type="password",
                                       placeholder="Repeat new password",
                                       key="sec_tab_new2")
                password_rules_card(v_new1)
                with st.form("prof_security_tab"):
                    ss_btn = st.form_submit_button(
                        "Change Password", use_container_width=True, type="primary")

            if ss_btn:
                if not v_old or not v_new1 or not v_new2:
                    st.error("All three fields are required.")
                elif v_new1 != v_new2:
                    st.error("New passwords do not match.")
                else:
                    ok, msg = change_password(uname, v_old, v_new1)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


    # ════ TAB 7: SUPPORT CHAT ═══════════════════════════════════════════════
    with tab7:
        mark_chat_read(uname, "patient")
        st.session_state._pat_unread_seen = 0
        msgs = get_chat_messages(uname)

        st.markdown(
            f'<div style="background:{t["p_light"]};border:1px solid {t["border"]};'
            f'border-radius:14px;padding:12px 18px;margin-bottom:14px;'
            f'display:flex;align-items:center;gap:10px;">'
            f'<span style="font-size:1.2rem;">💬</span>'
            f'<span style="font-size:.83rem;color:{t["t2"]};">'
            f'Send a message to the hospital admin. We typically reply within 24 hours.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        chat_box = st.container(height=430, border=False)
        with chat_box:
            if not msgs:
                st.markdown(
                    '<div style="text-align:center;padding:50px 0;color:#9CA3AF;font-size:.84rem;">'
                    '👋 No messages yet — say hello below!</div>',
                    unsafe_allow_html=True,
                )
            for m in msgs:
                is_mine = m["sender"] == "patient"
                st.markdown(
                    _chat_bubble_html(m["body"], is_mine, m.get("sent_at")),
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div style="border-top:1px solid #E5E7EB;margin-bottom:4px;"></div>',
            unsafe_allow_html=True,
        )
        body = st.chat_input("Type your message…", key="patient_chat_input")
        if body:
            send_chat_message(uname, "patient", body)
            st.rerun()


    # ════ TAB 6: MY BOOKINGS ════════════════════════════════════════════════
    with tab6:
        # Reschedule booking dialog trigger (set by clicking 🔄 on a card)
        if st.session_state.get("_open_reschedule"):
            _est_rs = st.session_state.pop("_reschedule_est", {})
            st.session_state["_open_reschedule"] = False
            if _est_rs:
                _booking_dialog(_est_rs, profile, t)

        bookings = list_patient_bookings(uname)

        STATUS_STYLE = {
            "pending":   ("#FFFBEB", "#D97706", "⏳ Pending"),
            "confirmed": ("#D1FAE5", "#059669", "✅ Confirmed"),
            "cancelled": ("#FEE2E2", "#DC2626", "❌ Cancelled"),
            "completed": ("#EFF6FF", "#2563EB", "🏁 Completed"),
            "no_show":   ("#F3F4F6", "#6B7280", "🚫 No Show"),
        }

        ACTIVE_STATUSES  = {"pending", "confirmed"}
        HISTORY_STATUSES = {"cancelled", "completed", "no_show"}

        active  = [b for b in bookings if b["status"] in ACTIVE_STATUSES]
        history = [b for b in bookings if b["status"] in HISTORY_STATUSES]

        def _booking_est_from(b: dict) -> dict:
            return {
                "selected_hosp_name": b.get("hospital_name", ""),
                "hospital_id":        b.get("hospital_id", ""),
                "department":         b.get("department", ""),
                "severity":           b.get("severity", ""),
                "admission_type":     b.get("admission_type", ""),
                "los_low":            b.get("los_low", 0),
                "los_high":           b.get("los_high", 0),
                "cost_low":           b.get("cost_low", 0),
                "cost_high":          b.get("cost_high", 0),
            }

        def _render_booking_card(b, allow_cancel=False, allow_reschedule=False, card_idx=0):
            bg, col, lbl = STATUS_STYLE.get(b["status"], STATUS_STYLE["pending"])
            adm_note   = b.get("admin_note", "")
            created    = b["created_at"].strftime("%d %b %Y") if b.get("created_at") else ""
            ref        = b["booking_ref"]
            cancel_key = f"_cancel_confirm_{ref}"

            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
                    f'<div>'
                    f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:800;'
                    f'font-size:.95rem;color:#111827;margin-bottom:3px;">'
                    f'{b.get("hospital_name","—")} · {b.get("department","—")}</div>'
                    f'<div style="font-size:.78rem;color:#6B7280;">'
                    f'📅 {b.get("requested_date","—")} &nbsp;·&nbsp; '
                    f'🕐 {b.get("preferred_time","—")} &nbsp;·&nbsp; '
                    f'Ref: <strong>{ref}</strong></div>'
                    f'</div>'
                    f'<span style="background:{bg};color:{col};border-radius:9999px;'
                    f'padding:4px 14px;font-size:.75rem;font-weight:700;white-space:nowrap;">'
                    f'{lbl}</span>'
                    f'</div>'
                    f'<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:6px;">'
                    f'<span style="font-size:.78rem;color:#4B5563;">'
                    f'🛏️ <strong>{int(b["los_low"])}–{int(b["los_high"])} days</strong></span>'
                    f'<span style="font-size:.78rem;color:#4B5563;">'
                    f'💰 <strong>Rs. {int(b["cost_low"]):,}–{int(b["cost_high"]):,}</strong></span>'
                    f'<span style="font-size:.78rem;color:#9CA3AF;">Booked: {created}</span>'
                    f'</div>'
                    + (f'<div style="background:#FEF2F2;border-radius:8px;'
                       f'padding:7px 10px;font-size:.76rem;color:#991B1B;margin-bottom:4px;">'
                       f'ℹ️ {adm_note}</div>' if adm_note else ""),
                    unsafe_allow_html=True,
                )

                # Action buttons row — right-aligned inside the card
                if allow_cancel or allow_reschedule:
                    if st.session_state.get(cancel_key):
                        # Inline cancel confirmation
                        st.warning(f"Cancel **{ref}**? This cannot be undone.")
                        yes_c, no_c, _ = st.columns([1, 1, 6])
                        with yes_c:
                            if st.button("Yes, Cancel",
                                         key=f"_cyes_{ref}_{card_idx}",
                                         type="primary",
                                         use_container_width=True):
                                update_booking_status(ref, "cancelled",
                                                      admin_note="Cancelled by patient")
                                st.session_state[cancel_key] = False
                                st.rerun()
                        with no_c:
                            if st.button("Keep it",
                                         key=f"_cno_{ref}_{card_idx}",
                                         use_container_width=True):
                                st.session_state[cancel_key] = False
                                st.rerun()
                    else:
                        # Icon buttons: 🔄 reschedule  ✕ cancel
                        spacer, rs_col, x_col = st.columns([10, 1, 1])
                        if allow_reschedule:
                            with rs_col:
                                if st.button("🔄",
                                             key=f"_rs_{ref}_{card_idx}",
                                             help="Reschedule this booking",
                                             use_container_width=True):
                                    st.session_state["_reschedule_est"]  = _booking_est_from(b)
                                    st.session_state["_open_reschedule"] = True
                                    st.rerun()
                        if allow_cancel:
                            with x_col:
                                if st.button("✕",
                                             key=f"_cico_{ref}_{card_idx}",
                                             help="Cancel this booking",
                                             use_container_width=True):
                                    st.session_state[cancel_key] = True
                                    st.rerun()

        if not bookings:
            st.markdown(
                f'<div style="text-align:center;padding:60px 0;">'
                f'<div style="font-size:3rem;margin-bottom:12px;">📅</div>'
                f'<div style="font-weight:700;color:{t["t1"]};font-size:1rem;margin-bottom:6px;">'
                f'No bookings yet</div>'
                f'<div style="font-size:.84rem;color:{t["t2"]};">'
                f'Get your estimate in the 💰 My Estimate tab, then click "Book This Visit".</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            bk_cur_tab, bk_hist_tab = st.tabs([
                f"📌  Current  ({len(active)})",
                f"🗂️  History  ({len(history)})",
            ])

            with bk_cur_tab:
                if not active:
                    st.markdown(
                        '<div style="text-align:center;padding:40px 0;color:#9CA3AF;">'
                        '<div style="font-size:2rem;margin-bottom:8px;">🎉</div>'
                        'No active bookings right now.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for i, b in enumerate(active):
                        _render_booking_card(b, allow_cancel=True,
                                             allow_reschedule=True, card_idx=i)

            with bk_hist_tab:
                if not history:
                    st.markdown(
                        '<div style="text-align:center;padding:40px 0;color:#9CA3AF;">'
                        '<div style="font-size:2rem;margin-bottom:8px;">📂</div>'
                        'No past bookings yet.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for i, b in enumerate(history):
                        _render_booking_card(b, allow_cancel=False,
                                             allow_reschedule=False, card_idx=i)


# ════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def _admin_patient_row(p: dict, i: int, t: dict):
    """Render one patient row card + delete + details expander."""
    uname   = p.get("username", "")
    pid     = p.get("patient_id", "—")
    name    = p.get("full_name") or "(no name set)"
    joined  = ""
    ca = p.get("created_at")
    if ca and hasattr(ca, "strftime"):
        joined = ca.strftime("%d %b %Y")
    is_sso   = bool(p.get("sso_provider"))
    display  = p.get("sso_email") or uname
    type_bg  = "#EFF6FF" if is_sso else "#F0FDF4"
    type_col = "#1D4ED8" if is_sso else "#166534"
    type_lbl = "Google SSO" if is_sso else "Password"
    initials = name[:2].upper() if name != "(no name set)" else "??"
    safe_name    = str(name).replace("<", "&lt;").replace(">", "&gt;")
    safe_display = str(display).replace("<", "&lt;").replace(">", "&gt;")

    st.markdown(
        f'<div style="background:#fff;border:1px solid #E5E7EB;border-radius:14px;'
        f'padding:16px 20px;margin-bottom:4px;display:flex;align-items:center;gap:16px;">'
        f'<div style="width:42px;height:42px;border-radius:50%;flex-shrink:0;'
        f'background:linear-gradient(135deg,{t["secondary"]},{t["primary"]});'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-weight:700;font-size:.82rem;color:#fff;">{initials}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:700;'
        f'font-size:.92rem;color:#111827;margin-bottom:2px;">{safe_name}</div>'
        f'<div style="font-size:.76rem;color:#6B7280;">{safe_display}</div>'
        f'</div>'
        f'<span style="background:#EDE9FE;color:#5B21B6;border-radius:9999px;'
        f'padding:3px 10px;font-size:.7rem;font-weight:700;flex-shrink:0;">{pid}</span>'
        f'<span style="background:{type_bg};color:{type_col};border-radius:9999px;'
        f'padding:3px 10px;font-size:.7rem;font-weight:600;flex-shrink:0;">{type_lbl}</span>'
        f'<div style="font-size:.76rem;color:#9CA3AF;flex-shrink:0;">{joined}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    act_col, detail_col = st.columns([1, 6])
    with act_col:
        if st.button("🗑️ Delete", key=f"del_btn_{i}", use_container_width=True):
            st.session_state.admin_del_target = uname

    if st.session_state.get("admin_del_target") == uname:
        st.warning(
            f"Permanently delete **{name}** (`{uname}`)? "
            f"This removes their account, profile, chat history, and all session data."
        )
        yes_col, no_col, _ = st.columns([1, 1, 4])
        with yes_col:
            if st.button("Yes, Delete", key=f"confirm_{i}",
                         type="primary", use_container_width=True):
                ok, msg = delete_patient_account(uname)
                st.session_state.admin_del_target = None
                st.success(msg) if ok else st.error(msg)
                st.rerun()
        with no_col:
            if st.button("Cancel", key=f"cancel_{i}", use_container_width=True):
                st.session_state.admin_del_target = None
                st.rerun()

    with detail_col:
        with st.expander("View details", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**District:** {p.get('district') or '—'}")
            d1.markdown(f"**Gender:** {p.get('gender') or '—'}")
            d2.markdown(f"**Blood type:** {p.get('blood_type') or '—'}")
            d2.markdown(f"**Chronic conditions:** {p.get('chronic_conditions') or 'None'}")
            d3.markdown(f"**Preferred hospital:** {p.get('pref_hospital') or '—'}")


def admin_dashboard():
    import datetime as _dt
    t = ADMIN_THEME
    inject_global_css(t)
    render_sidebar(t)

    # ── Load data needed for bell before rendering the bar ────────────────────
    patients  = list_patient_accounts()
    total     = len(patients)
    now       = _dt.datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    new_month   = sum(1 for p in patients
                      if p.get("created_at") and p["created_at"] >= month_start)
    inbox        = get_chat_inbox()
    unread_total = sum(th.get("unread", 0) for th in inbox)

    pending_bookings = list_all_bookings(status_filter="pending")
    admin_notifs = [
        {
            "severity": "urgent",
            "icon":     "📅",
            "title":    f"New booking from {b['full_name']}",
            "body":     f"{b['hospital_name']} · {b['department']} · {b['requested_date']} — Ref: {b['booking_ref']}",
        }
        for b in pending_bookings
    ] + [
        {
            "severity": "warning" if th.get("unread", 0) > 0 else "info",
            "icon":     "💬",
            "title":    f"New message from {th['full_name']}",
            "body":     (th.get("latest_body") or "")[:70],
        }
        for th in inbox if th.get("unread", 0) > 0
    ]
    render_session_bar(notifs=admin_notifs, n_total=len(admin_notifs))

    st.markdown(welcome_banner(
        "Admin Console",
        "Bagmati Hospital Intelligence System · Patient Management & Messaging",
        t
    ), unsafe_allow_html=True)

    # ── Toast when new patient messages arrive ────────────────────────────────
    if "_adm_unread_seen" not in st.session_state:
        st.session_state._adm_unread_seen = unread_total
    elif unread_total > st.session_state._adm_unread_seen:
        diff = unread_total - st.session_state._adm_unread_seen
        st.toast(f"💬 {diff} new patient message{'s' if diff > 1 else ''}!", icon="🔔")
        st.session_state._adm_unread_seen = unread_total

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(t["p_light"], "👥", str(total),
                         "Total Patients", "All registered accounts", t), unsafe_allow_html=True)
    c2.markdown(kpi_card("#EFF6FF", "🆕", str(new_month),
                         "New This Month", now.strftime("%B %Y"), t), unsafe_allow_html=True)
    c3.markdown(kpi_card("#F0FDF4", "💬", str(len(inbox)),
                         "Active Conversations", "Patients who messaged", t), unsafe_allow_html=True)
    c4.markdown(kpi_card("#FEF2F2", "🔴", str(unread_total),
                         "Unread Messages", "Waiting for your reply", t), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    _msg_label = f"💬  Messages {'🔴' if unread_total else ''}"
    _pending_count = len(pending_bookings)
    _bk_label  = f"📅  Bookings {'🔴 ' + str(_pending_count) if _pending_count else ''}"
    admin_tab4, admin_tab5, admin_tab6, admin_tab1, admin_tab3, admin_tab2 = st.tabs([
        "📊  Analytics", "🏥  Hospital Data", "🎯  Model Accuracy",
        "👥  Patient Accounts", _bk_label, _msg_label
    ])

    # Shared chart helpers/palette (used by Analytics, Hospital Data, Model Accuracy)
    GREEN  = "#10B981"; AMBER = "#F59E0B"; RED   = "#EF4444"
    BLUE   = "#3B82F6"; GRAY  = "#9CA3AF"; INDIGO = "#6366F1"
    PURPLE = "#7C3AED"; TEAL  = "#0D9488"
    BG     = "rgba(0,0,0,0)"

    def _base(fig, title="", legend=False):
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color="#374151",
                       family="Plus Jakarta Sans"), x=0),
            paper_bgcolor=BG, plot_bgcolor=BG,
            margin=dict(l=8, r=8, t=44, b=8),
            font=dict(family="Inter", size=11, color="#6B7280"),
            showlegend=legend,
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10)),
        )
        fig.update_xaxes(showgrid=False, zeroline=False,
                         linecolor="#E5E7EB", tickfont=dict(size=10))
        fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6",
                         zeroline=False, tickfont=dict(size=10))
        return fig

    # ════ TAB 1: PATIENT ACCOUNTS ═══════════════════════════════════════════
    with admin_tab1:
        if "admin_del_target" not in st.session_state:
            st.session_state.admin_del_target = None

        search = st.text_input(
            "Search", placeholder="Patient ID, name, or username…",
            label_visibility="collapsed",
        )
        filtered = patients
        if search.strip():
            q = search.strip().lower()
            filtered = [p for p in patients if
                        q in (p.get("patient_id") or "").lower() or
                        q in (p.get("full_name")  or "").lower() or
                        q in (p.get("username")   or "").lower()]

        st.markdown(
            f'<div style="font-size:.78rem;color:#6B7280;margin-bottom:12px;">'
            f'{len(filtered)} account{"s" if len(filtered) != 1 else ""}</div>',
            unsafe_allow_html=True,
        )

        if not filtered:
            st.info("No patient accounts found.")
        else:
            for i, p in enumerate(filtered):
                _admin_patient_row(p, i, t)

    # ════ TAB 2: MESSAGES ═══════════════════════════════════════════════════
    with admin_tab2:
        if "admin_chat_sel" not in st.session_state:
            st.session_state.admin_chat_sel = None

        if not inbox:
            st.info("No patient messages yet.")
        else:
            left_col, right_col = st.columns([1, 2.2], gap="large")

            with left_col:
                # Inject marker + CSS that styles all sibling buttons as conversation cards
                st.markdown(
                    '<div class="conv-list-marker"></div>'
                    '<div style="font-size:.75rem;font-weight:700;color:#6B7280;'
                    'text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">'
                    'Conversations</div>'
                    '<style>'
                    # All buttons after the marker → white card style (override global red)
                    '[data-testid="stMarkdownContainer"]:has(.conv-list-marker)'
                    ' ~ div [data-testid="stButton"]>button{'
                    '  background:#fff!important;color:#111827!important;'
                    '  border:1px solid #E5E7EB!important;border-radius:12px!important;'
                    '  text-align:left!important;padding:10px 13px!important;'
                    '  font-size:.86rem!important;font-weight:700!important;'
                    '  height:auto!important;white-space:pre-line!important;'
                    '  line-height:1.55!important;box-shadow:none!important;'
                    '  letter-spacing:0!important;width:100%!important;'
                    '  transition:all .15s!important;'
                    '}'
                    '[data-testid="stMarkdownContainer"]:has(.conv-list-marker)'
                    ' ~ div [data-testid="stButton"]>button:hover{'
                    '  border-color:#10B981!important;'
                    '  box-shadow:0 2px 10px rgba(16,185,129,.15)!important;'
                    '  transform:translateY(-1px)!important;'
                    '}'
                    # Selected conversation = primary button → green tint card
                    '[data-testid="stMarkdownContainer"]:has(.conv-list-marker)'
                    ' ~ div [data-testid="stButton"]>button[kind="primary"]{'
                    '  background:#F0FDF4!important;border:2px solid #10B981!important;'
                    '  box-shadow:none!important;transform:none!important;'
                    '}'
                    '</style>',
                    unsafe_allow_html=True,
                )

                for th in inbox:
                    _cu        = th["patient_username"]
                    _cname     = th["full_name"]
                    _cunread   = th.get("unread", 0)
                    _cpreview  = (th.get("latest_body") or "")[:38]
                    _cts       = th.get("latest_at")
                    _cts_str   = _cts.strftime("%d %b") if _cts and hasattr(_cts, "strftime") else ""
                    _is_sel    = st.session_state.admin_chat_sel == _cu

                    _unread_str = f"  🔴 {_cunread}" if _cunread else ""
                    _label      = f"{_cname}{_unread_str}"
                    _btn_type   = "primary" if _is_sel else "secondary"

                    if st.button(_label, key=f"open_chat_{_cu}",
                                 type=_btn_type, use_container_width=True):
                        st.session_state.admin_chat_sel = _cu
                        st.rerun()

            with right_col:
                sel = st.session_state.admin_chat_sel
                if not sel:
                    st.markdown(
                        '<div style="text-align:center;padding:80px 0;color:#9CA3AF;font-size:.86rem;">'
                        '← Select a conversation</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    mark_chat_read(sel, "admin")
                    st.session_state._adm_unread_seen = sum(
                        th.get("unread", 0) for th in get_chat_inbox()
                    )
                    sel_name = next(
                        (th["full_name"] for th in inbox if th["patient_username"] == sel), sel
                    )
                    safe_sel_name = str(sel_name).replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(
                        f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:700;'
                        f'font-size:.98rem;color:{t["t1"]};padding-bottom:10px;'
                        f'border-bottom:1px solid #E5E7EB;margin-bottom:10px;">'
                        f'💬 &nbsp;{safe_sel_name}</div>',
                        unsafe_allow_html=True,
                    )

                    msgs = get_chat_messages(sel)
                    chat_box = st.container(height=390, border=False)
                    with chat_box:
                        if not msgs:
                            st.markdown(
                                '<div style="text-align:center;padding:50px 0;'
                                'color:#9CA3AF;font-size:.84rem;">No messages yet.</div>',
                                unsafe_allow_html=True,
                            )
                        for m in msgs:
                            # From admin's view: admin = right (is_mine), patient = left
                            is_mine = m["sender"] == "admin"
                            st.markdown(
                                _chat_bubble_html(m["body"], is_mine, m.get("sent_at")),
                                unsafe_allow_html=True,
                            )

                    st.markdown(
                        '<div style="border-top:1px solid #E5E7EB;margin-bottom:4px;"></div>',
                        unsafe_allow_html=True,
                    )
                    reply = st.chat_input("Reply as admin…", key="admin_chat_input")
                    if reply:
                        send_chat_message(sel, "admin", reply)
                        st.rerun()

    # ════ TAB 3: BOOKINGS ═══════════════════════════════════════════════════
    with admin_tab3:
        STATUS_STYLE = {
            "pending":   ("#FFFBEB", "#D97706", "⏳ Pending"),
            "confirmed": ("#D1FAE5", "#059669", "✅ Confirmed"),
            "cancelled": ("#FEE2E2", "#DC2626", "❌ Cancelled"),
            "completed": ("#EFF6FF", "#2563EB", "🏁 Completed"),
            "no_show":   ("#F3F4F6", "#6B7280", "🚫 No Show"),
        }

        if "admin_bk_action" not in st.session_state:
            st.session_state.admin_bk_action = {}

        all_bk  = list_all_bookings()
        history = [b for b in all_bk if b["status"] in ("cancelled", "completed", "no_show")]

        def _admin_bk_card(b, card_key_prefix=""):
            bg, col, lbl = STATUS_STYLE.get(b["status"], STATUS_STYLE["pending"])
            ref     = b["booking_ref"]
            created = b["created_at"].strftime("%d %b %Y") if b.get("created_at") else ""

            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'flex-wrap:wrap;gap:8px;margin-bottom:8px;">'
                    f'<div>'
                    f'<div style="font-weight:700;font-size:.92rem;color:#111827;margin-bottom:3px;">'
                    f'{b.get("full_name","—")} &nbsp;'
                    f'<span style="background:#EDE9FE;color:#5B21B6;border-radius:9999px;'
                    f'padding:1px 8px;font-size:.68rem;">{b.get("patient_id","")}</span></div>'
                    f'<div style="font-size:.77rem;color:#6B7280;">'
                    f'{b.get("hospital_name","—")} · {b.get("department","—")} · '
                    f'{b.get("requested_date","—")} &nbsp;{b.get("preferred_time","")}</div>'
                    f'<div style="font-size:.72rem;color:#9CA3AF;margin-top:2px;">'
                    f'Ref: <strong>{ref}</strong> · Submitted: {created} · '
                    f'📞 {b.get("phone","—")}</div>'
                    f'</div>'
                    f'<span style="background:{bg};color:{col};border-radius:9999px;'
                    f'padding:4px 14px;font-size:.74rem;font-weight:700;">{lbl}</span>'
                    f'</div>'
                    f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:6px;">'
                    f'<span style="font-size:.76rem;color:#4B5563;">'
                    f'🛏️ <strong>{int(b["los_low"])}–{int(b["los_high"])} days</strong></span>'
                    f'<span style="font-size:.76rem;color:#4B5563;">'
                    f'💰 <strong>Rs. {int(b["cost_low"]):,}–{int(b["cost_high"]):,}</strong></span>'
                    f'<span style="font-size:.76rem;color:#4B5563;">'
                    f'{b.get("severity","—")} severity · {b.get("admission_type","—")}</span>'
                    f'</div>'
                    + (f'<div style="font-size:.76rem;color:#6B7280;background:#F9FAFB;'
                       f'border-radius:8px;padding:6px 10px;margin-bottom:4px;">'
                       f'📝 {b["notes"]}</div>' if b.get("notes") else "")
                    + (f'<div style="font-size:.76rem;color:#991B1B;background:#FEF2F2;'
                       f'border-radius:8px;padding:6px 10px;margin-bottom:4px;">'
                       f'ℹ️ {b["admin_note"]}</div>' if b.get("admin_note") else ""),
                    unsafe_allow_html=True,
                )

                if b["status"] == "pending":
                    a1, a2, _ = st.columns([1, 1, 4])
                    with a1:
                        if st.button("✅ Confirm", key=f"{card_key_prefix}confirm_{ref}",
                                     use_container_width=True, type="primary"):
                            update_booking_status(ref, "confirmed")
                            st.success(f"{ref} confirmed.")
                            st.rerun()
                    with a2:
                        if st.button("❌ Cancel", key=f"{card_key_prefix}cancel_{ref}",
                                     use_container_width=True):
                            st.session_state.admin_bk_action[ref] = "cancel"
                            st.rerun()

                if st.session_state.admin_bk_action.get(ref) == "cancel":
                    reason = st.text_input(f"Reason for cancelling {ref}",
                                           key=f"{card_key_prefix}reason_{ref}",
                                           placeholder="e.g. slot unavailable")
                    r1, r2, _ = st.columns([1, 1, 4])
                    with r1:
                        if st.button("Confirm Cancel",
                                     key=f"{card_key_prefix}do_cancel_{ref}",
                                     type="primary", use_container_width=True):
                            update_booking_status(ref, "cancelled", reason)
                            st.session_state.admin_bk_action.pop(ref, None)
                            st.success(f"{ref} cancelled.")
                            st.rerun()
                    with r2:
                        if st.button("Go Back",
                                     key=f"{card_key_prefix}back_{ref}",
                                     use_container_width=True):
                            st.session_state.admin_bk_action.pop(ref, None)
                            st.rerun()

        pending_bk   = [b for b in all_bk if b["status"] == "pending"]
        confirmed_bk = [b for b in all_bk if b["status"] == "confirmed"]

        adm_pend_tab, adm_conf_tab, adm_hist_tab = st.tabs([
            f"⏳  Pending  ({len(pending_bk)})",
            f"✅  Approved  ({len(confirmed_bk)})",
            f"🗂️  History  ({len(history)})",
        ])

        with adm_pend_tab:
            if not pending_bk:
                st.info("No pending requests.")
            else:
                for b in pending_bk:
                    _admin_bk_card(b, card_key_prefix="pend_")

        with adm_conf_tab:
            if not confirmed_bk:
                st.info("No confirmed bookings yet.")
            else:
                for b in confirmed_bk:
                    _admin_bk_card(b, card_key_prefix="conf_")

        with adm_hist_tab:
            if not history:
                st.info("No past bookings yet.")
            else:
                for b in history:
                    _admin_bk_card(b, card_key_prefix="hist_")

    # ════ TAB 4: ANALYTICS ══════════════════════════════════════════════════
    with admin_tab4:
        _all_bk_a   = list_all_bookings()
        _patients_a = patients

        # ── Row 1 ─────────────────────────────────────────────────────────────
        r1c1, r1c2 = st.columns(2)

        # 1. FUNNEL — booking pipeline (pending → confirmed → completed)
        with r1c1:
            sc = {b["status"]: 0 for b in _all_bk_a}
            for b in _all_bk_a:
                sc[b["status"]] += 1
            funnel_order  = ["pending", "confirmed", "completed",
                             "cancelled", "no_show"]
            funnel_labels = [s.capitalize() for s in funnel_order
                             if s in sc]
            funnel_vals   = [sc[s] for s in funnel_order if s in sc]
            funnel_colors = [{"pending": AMBER, "confirmed": GREEN,
                              "completed": BLUE, "cancelled": RED,
                              "no_show": GRAY}[s]
                             for s in funnel_order if s in sc]

            fig_funnel = go.Figure(go.Funnel(
                y=funnel_labels, x=funnel_vals,
                textinfo="value+percent initial",
                marker=dict(color=funnel_colors,
                            line=dict(color="#fff", width=1)),
                connector=dict(line=dict(color="#E5E7EB", width=1)),
                hovertemplate="%{y}: %{x}<extra></extra>",
            ))
            _base(fig_funnel, "Booking Pipeline (Funnel)")
            st.plotly_chart(fig_funnel, use_container_width=True)

        # 2. TREEMAP — bookings by department
        with r1c2:
            dept_map = {}
            for b in _all_bk_a:
                d = b.get("department", "Unknown")
                dept_map[d] = dept_map.get(d, 0) + 1

            if dept_map:
                fig_tree = go.Figure(go.Treemap(
                    labels=list(dept_map.keys()),
                    parents=[""] * len(dept_map),
                    values=list(dept_map.values()),
                    textinfo="label+value",
                    marker=dict(
                        colorscale=[[0, "#D1FAE5"], [0.5, "#34D399"],
                                    [1, "#059669"]],
                        showscale=False,
                        line=dict(color="#fff", width=2),
                    ),
                    hovertemplate="%{label}: %{value} bookings<extra></extra>",
                ))
                _base(fig_tree, "Department Demand (Treemap)")
                fig_tree.update_layout(margin=dict(l=4, r=4, t=44, b=4))
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("No department data yet.")

        # ── Row 2 ─────────────────────────────────────────────────────────────
        r2c1, r2c2 = st.columns(2)

        # 3. STEP AREA — daily bookings over time (staircase, suits discrete counts)
        with r2c1:
            date_map = {}
            for b in _all_bk_a:
                if b.get("created_at"):
                    day = b["created_at"].strftime("%Y-%m-%d")
                    date_map[day] = date_map.get(day, 0) + 1

            if date_map:
                sorted_d = sorted(date_map.items())
                step_x = [d[0] for d in sorted_d]
                step_y = [d[1] for d in sorted_d]

                fig_step = go.Figure(go.Scatter(
                    x=step_x, y=step_y,
                    mode="lines+markers",
                    line=dict(shape="hv", color=GREEN, width=2.5),
                    marker=dict(color=GREEN, size=7, symbol="circle",
                                line=dict(color="#fff", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(16,185,129,0.08)",
                    hovertemplate="%{x}: %{y} bookings<extra></extra>",
                ))
                _base(fig_step, "Daily Booking Requests (Step Chart)")
                fig_step.update_layout(showlegend=False)
                st.plotly_chart(fig_step, use_container_width=True)
            else:
                st.info("No timeline data yet.")

        # 4. SCATTER BUBBLE — hospital: x=total, y=pending, size=total (meaningful axes)
        with r2c2:
            hosp_data = {}
            for b in _all_bk_a:
                h = b.get("hospital_name", "Unknown")
                if h not in hosp_data:
                    hosp_data[h] = {"total": 0, "pending": 0, "confirmed": 0}
                hosp_data[h]["total"] += 1
                if b["status"] == "pending":
                    hosp_data[h]["pending"] += 1
                elif b["status"] == "confirmed":
                    hosp_data[h]["confirmed"] += 1

            if hosp_data:
                h_names     = list(hosp_data.keys())
                h_totals    = [hosp_data[h]["total"]     for h in h_names]
                h_pending   = [hosp_data[h]["pending"]   for h in h_names]
                h_confirmed = [hosp_data[h]["confirmed"] for h in h_names]

                fig_bub = go.Figure(go.Scatter(
                    x=h_totals,
                    y=h_pending,
                    mode="markers+text",
                    marker=dict(
                        size=[max(t * 14, 22) for t in h_totals],
                        color=h_confirmed,
                        colorscale=[[0, "#D1FAE5"], [1, "#059669"]],
                        showscale=True,
                        colorbar=dict(title="Confirmed", thickness=10,
                                      len=0.7, tickfont=dict(size=9)),
                        line=dict(color="#fff", width=2),
                        opacity=0.85,
                    ),
                    text=h_names,
                    textposition="top center",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Total bookings: %{x}<br>"
                        "Pending: %{y}<br>"
                        "Confirmed: %{marker.color}<extra></extra>"
                    ),
                ))
                _base(fig_bub, "Hospital Load — Total vs Pending (Bubble)")
                fig_bub.update_xaxes(title_text="Total Bookings",
                                     title_font=dict(size=10))
                fig_bub.update_yaxes(title_text="Pending Bookings",
                                     title_font=dict(size=10))
                st.plotly_chart(fig_bub, use_container_width=True)
            else:
                st.info("No hospital data yet.")

        # ── Row 3 ─────────────────────────────────────────────────────────────
        r3c1, r3c2 = st.columns(2)

        # 5. SUNBURST — severity (inner) × booking status (outer)
        with r3c1:
            from collections import defaultdict as _dd
            sev_status_map = _dd(lambda: _dd(int))
            for b in _all_bk_a:
                sev = b.get("severity", "Unknown")
                sts = b["status"].capitalize()
                sev_status_map[sev][sts] += 1

            if sev_status_map:
                sev_order = ["Mild", "Moderate", "Severe", "Critical", "Unknown"]
                sev_color_map = {
                    "Mild": GREEN, "Moderate": AMBER,
                    "Severe": RED, "Critical": PURPLE, "Unknown": GRAY,
                }
                sun_ids, sun_labels, sun_parents, sun_vals, sun_colors = \
                    [], [], [], [], []

                for sev in sev_order:
                    if sev not in sev_status_map:
                        continue
                    total = sum(sev_status_map[sev].values())
                    sun_ids.append(sev)
                    sun_labels.append(sev)
                    sun_parents.append("")
                    sun_vals.append(total)
                    sun_colors.append(sev_color_map.get(sev, GRAY))
                    for sts, cnt in sev_status_map[sev].items():
                        sun_ids.append(f"{sev}_{sts}")
                        sun_labels.append(sts)
                        sun_parents.append(sev)
                        sun_vals.append(cnt)
                        sun_colors.append(sev_color_map.get(sev, GRAY))

                fig_sun = go.Figure(go.Sunburst(
                    ids=sun_ids,
                    labels=sun_labels,
                    parents=sun_parents,
                    values=sun_vals,
                    marker=dict(colors=sun_colors,
                                line=dict(color="#fff", width=1.5)),
                    textinfo="label+value",
                    hovertemplate=(
                        "%{label}<br>Count: %{value}<br>"
                        "Share: %{percentRoot:.1%}<extra></extra>"
                    ),
                    insidetextorientation="radial",
                    maxdepth=2,
                ))
                _base(fig_sun, "Severity × Outcome (Sunburst)")
                fig_sun.update_layout(margin=dict(l=4, r=4, t=44, b=4))
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.info("No severity data yet.")

        # 6. HEATMAP — dept × status cross matrix
        with r3c2:
            depts_u = sorted({b.get("department", "?") for b in _all_bk_a})
            statuses_u = ["pending", "confirmed", "completed",
                          "cancelled", "no_show"]

            if depts_u and _all_bk_a:
                matrix = [[0] * len(statuses_u) for _ in depts_u]
                for b in _all_bk_a:
                    di = depts_u.index(b.get("department", "?")) \
                         if b.get("department", "?") in depts_u else 0
                    si = statuses_u.index(b["status"]) \
                         if b["status"] in statuses_u else 0
                    matrix[di][si] += 1

                fig_hm = go.Figure(go.Heatmap(
                    z=matrix,
                    x=[s.capitalize() for s in statuses_u],
                    y=depts_u,
                    colorscale=[[0, "#F0FDF4"], [0.5, "#34D399"],
                                [1, "#065F46"]],
                    text=[[str(v) for v in row] for row in matrix],
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    hovertemplate=(
                        "Dept: %{y}<br>Status: %{x}<br>"
                        "Count: %{z}<extra></extra>"
                    ),
                    showscale=True,
                    colorbar=dict(thickness=10, len=0.7,
                                  tickfont=dict(size=9)),
                ))
                _base(fig_hm, "Dept × Status Heatmap")
                fig_hm.update_yaxes(showgrid=False)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("No cross-data available yet.")

    # ════ TAB 5: HOSPITAL DATA (real regional dataset, read-only) ═══════════
    with admin_tab5:
        _adm_a = load_admissions()
        _occ_a = load_occupancy()

        st.markdown(
            '<div style="font-size:.78rem;color:#6B7280;margin-bottom:14px;">'
            'Computed live from the regional admissions &amp; occupancy dataset '
            '(2021–2024, 5 hospitals) — read-only, the source files are never modified.</div>',
            unsafe_allow_html=True,
        )

        _total_adm = len(_adm_a)
        _avg_los   = _adm_a["length_of_stay_days"].mean()
        _avg_cost  = _adm_a["total_bill_npr"].mean()
        _avg_occ   = _occ_a["occupancy_rate"].mean()
        _recovery  = (_adm_a["discharge_outcome"] == "Recovered").mean()
        _readmit   = _adm_a["readmission_flag"].mean()

        hk1, hk2, hk3, hk4, hk5, hk6 = st.columns(6)
        hk1.markdown(kpi_card(t["p_light"], "🛏️", f"{_total_adm:,}",
                     "Total Admissions", "2021–2024 dataset", t), unsafe_allow_html=True)
        hk2.markdown(kpi_card("#EFF6FF", "📆", f"{_avg_los:.1f}d",
                     "Avg. Length of Stay", f"WHO benchmark {config.WHO_LOS_DAYS}d", t), unsafe_allow_html=True)
        hk3.markdown(kpi_card("#F0FDF4", "💰", f"Rs {_avg_cost:,.0f}",
                     "Avg. Bill", "Per admission", t), unsafe_allow_html=True)
        hk4.markdown(kpi_card("#FFFBEB", "📊", f"{_avg_occ:.0%}",
                     "Avg. Occupancy", f"WHO surge {config.WHO_OCCUPANCY_THRESHOLD:.0%}", t), unsafe_allow_html=True)
        hk5.markdown(kpi_card("#ECFDF5", "✅", f"{_recovery:.0%}",
                     "Recovery Rate", "Discharge outcome", t), unsafe_allow_html=True)
        hk6.markdown(kpi_card("#FEF2F2", "🔁", f"{_readmit:.0%}",
                     "Readmission Rate", "Flagged returns", t), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        hd_r1c1, hd_r1c2 = st.columns(2)

        # Admissions by hospital
        with hd_r1c1:
            _by_hosp    = _adm_a.groupby("hospital_id").size().sort_values()
            _hosp_names = [config.HOSPITAL_NAMES.get(h, h) for h in _by_hosp.index]
            fig_h = go.Figure(go.Bar(
                x=_by_hosp.values, y=_hosp_names, orientation="h",
                marker=dict(color=TEAL),
                hovertemplate="%{y}: %{x:,} admissions<extra></extra>",
            ))
            _base(fig_h, "Admissions by Hospital")
            st.plotly_chart(fig_h, use_container_width=True)

        # Monthly occupancy trend vs WHO thresholds
        with hd_r1c2:
            _occ_trend = (_occ_a.assign(month=_occ_a["date"].dt.to_period("M").astype(str))
                                .groupby("month")["occupancy_rate"].mean().reset_index())
            fig_t = go.Figure(go.Scatter(
                x=_occ_trend["month"], y=_occ_trend["occupancy_rate"],
                mode="lines", line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            ))
            fig_t.add_hline(y=config.WHO_OCCUPANCY_THRESHOLD, line_dash="dash", line_color=RED,
                            annotation_text="WHO surge", annotation_font_size=10)
            fig_t.add_hline(y=config.WARN_OCCUPANCY_THRESHOLD, line_dash="dot", line_color=AMBER,
                            annotation_text="Warning", annotation_font_size=10)
            _base(fig_t, "Monthly Occupancy Trend")
            fig_t.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_t, use_container_width=True)

        hd_r2c1, hd_r2c2 = st.columns(2)

        # LOS distribution
        with hd_r2c1:
            fig_los = go.Figure(go.Histogram(
                x=_adm_a["length_of_stay_days"], nbinsx=30,
                marker=dict(color=INDIGO),
                hovertemplate="%{x} days: %{y:,} admissions<extra></extra>",
            ))
            _base(fig_los, "Length of Stay Distribution")
            st.plotly_chart(fig_los, use_container_width=True)

        # Severity mix
        with hd_r2c2:
            _sev_order  = ["Mild", "Moderate", "Severe", "Critical"]
            _sev_colors = {"Mild": GREEN, "Moderate": AMBER, "Severe": RED, "Critical": PURPLE}
            _sev        = _adm_a["severity"].value_counts().reindex(_sev_order).dropna()
            fig_sev = go.Figure(go.Pie(
                labels=_sev.index, values=_sev.values, hole=0.55,
                marker=dict(colors=[_sev_colors[s] for s in _sev.index]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            _base(fig_sev, "Admissions by Severity", legend=True)
            st.plotly_chart(fig_sev, use_container_width=True)

        hd_r3c1, hd_r3c2 = st.columns(2)

        # Department volume (top 10)
        with hd_r3c1:
            _dept = _adm_a["department_name"].value_counts().head(10).sort_values()
            fig_dept = go.Figure(go.Bar(
                x=_dept.values, y=_dept.index, orientation="h",
                marker=dict(color=PURPLE),
                hovertemplate="%{y}: %{x:,} admissions<extra></extra>",
            ))
            _base(fig_dept, "Top 10 Departments by Volume")
            st.plotly_chart(fig_dept, use_container_width=True)

        # Discharge outcomes
        with hd_r3c2:
            _out_colors = {"Recovered": GREEN, "Transferred": BLUE, "Referred": INDIGO,
                          "Expired": RED, "LAMA": AMBER}
            _out = _adm_a["discharge_outcome"].value_counts()
            fig_out = go.Figure(go.Bar(
                x=_out.index, y=_out.values,
                marker=dict(color=[_out_colors.get(o, GRAY) for o in _out.index]),
                hovertemplate="%{x}: %{y:,}<extra></extra>",
            ))
            _base(fig_out, "Discharge Outcomes")
            st.plotly_chart(fig_out, use_container_width=True)

    # ════ TAB 6: MODEL ACCURACY ═══════════════════════════════════════════════
    with admin_tab6:
        import datetime as _dt3

        st.markdown(
            '<div style="font-size:.78rem;color:#6B7280;margin-bottom:14px;">'
            'Live accuracy of the trained Random Forest models, measured on a held-out '
            '20% test split — recomputed each time <code>src/train_models.py</code> runs. '
            'Ranges shown to patients are point-prediction ± this MAE.</div>',
            unsafe_allow_html=True,
        )

        _metrics = model_metrics()

        if not _metrics:
            st.info("No trained models found. Run `python src/train_models.py` to train.")
        else:
            _los_m  = _metrics.get("los", {})
            _cost_m = _metrics.get("cost", {})
            _los_acc, _cost_acc = _los_m.get("accuracy_pct"), _cost_m.get("accuracy_pct")

            def _acc_gauge(value, title, bar_color):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    number={"suffix": "%", "font": {"size": 32, "color": "#111827"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9CA3AF",
                                "tickfont": {"size": 9}},
                        "bar": {"color": bar_color, "thickness": 0.28},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50],  "color": "#FEE2E2"},
                            {"range": [50, 75], "color": "#FEF3C7"},
                            {"range": [75, 100], "color": "#D1FAE5"},
                        ],
                    },
                ))
                _base(fig, title)
                fig.update_layout(height=210, margin=dict(l=20, r=20, t=44, b=10))
                return fig

            ga1, ga2 = st.columns(2)
            with ga1:
                if _los_acc is not None:
                    st.plotly_chart(_acc_gauge(_los_acc, "LOS Model — Accuracy", TEAL),
                                    use_container_width=True)
                else:
                    st.info("Retrain (`src/train_models.py`) to compute LOS accuracy.")
            with ga2:
                if _cost_acc is not None:
                    st.plotly_chart(_acc_gauge(_cost_acc, "Cost Model — Accuracy", INDIGO),
                                    use_container_width=True)
                else:
                    st.info("Retrain (`src/train_models.py`) to compute Cost accuracy.")

            st.markdown(
                '<div style="font-size:.7rem;color:#9CA3AF;margin:-6px 0 16px;">'
                'Accuracy = 100 − (Mean Abs. Error ÷ average actual value) × 100 — '
                'how close predictions land to real outcomes, on a held-out 20% test split.</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:700;'
                'font-size:.9rem;color:#111827;margin:6px 0 12px;">Other Thesis Models</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:.7rem;color:#9CA3AF;margin-top:-6px;margin-bottom:14px;">'
                'From the wider thesis analysis (Raw_data_outputs/ml_models/) — '
                'not served live by this app. (Length of Stay is omitted here — '
                'the LOS gauge above already covers it.) '
                'Readmission\'s 91% is misleading: precision is only 0.08 on the '
                'minority (positive) class.</div>',
                unsafe_allow_html=True,
            )

            _thesis_m = thesis_model_metrics()
            _order    = ["occupancy", "discharge", "overtime", "readmission"]
            _colors   = {"occupancy": TEAL, "discharge": INDIGO,
                        "overtime": AMBER, "readmission": RED}
            _items    = [(k, _thesis_m[k]) for k in _order if k in _thesis_m]

            if _items:
                _cols = st.columns(len(_items))
                for col, (key, it) in zip(_cols, _items):
                    with col:
                        st.plotly_chart(
                            _acc_gauge(it["accuracy_pct"], it["label"], _colors[key]),
                            use_container_width=True,
                        )
                        st.markdown(
                            f'<div style="font-size:.68rem;color:#9CA3AF;text-align:center;'
                            f'margin-top:-10px;">{it["detail"]}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Thesis model metrics not found in Raw_data_outputs/ml_models/.")

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            _trained_at = _los_m.get("trained_at")
            _trained_str = (_dt3.datetime.fromtimestamp(_trained_at).strftime("%d %b %Y, %H:%M")
                            if _trained_at else "—")
            st.markdown(
                '<div style="background:#fff;border:1px solid #E5E7EB;border-radius:14px;'
                'padding:14px 20px;font-size:.82rem;color:#374151;line-height:1.9;">'
                '<strong>LOS / Cost model details</strong>&nbsp;&nbsp;·&nbsp;&nbsp;'
                'Algorithm: RandomForestRegressor (scikit-learn)&nbsp;&nbsp;·&nbsp;&nbsp;'
                f'Trees: {_los_m.get("n_estimators", "—")}&nbsp;&nbsp;·&nbsp;&nbsp;'
                f'Max depth: {_los_m.get("max_depth", "—")}&nbsp;&nbsp;·&nbsp;&nbsp;'
                'Split: 80/20, random_state=42&nbsp;&nbsp;·&nbsp;&nbsp;'
                f'Last trained: {_trained_str}'
                '</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# LANDING PAGE — shown once per session, before login
# ════════════════════════════════════════════════════════════════════════════

def page_landing():
    st.markdown(
        "<style>#MainMenu, header, footer{visibility:hidden;}"
        ".stApp,[data-testid='stAppViewContainer']{background:#f5f3ee !important;}"
        ".block-container{padding:0 !important;max-width:100% !important;}"
        "[data-testid='stSidebar'],.stButton{display:none !important;}</style>",
        unsafe_allow_html=True,
    )
    landing_path = os.path.join(BASE_DIR, "static", "landing.html")
    with open(landing_path, encoding="utf-8") as f:
        landing_html = f.read()
    hero_path = os.path.join(BASE_DIR, "static", "hospital-hero.png")
    with open(hero_path, "rb") as f:
        hero_data = base64.b64encode(f.read()).decode("ascii")
    landing_html = landing_html.replace(
        "__HERO_IMAGE__", f"data:image/png;base64,{hero_data}"
    )
    st.html(landing_html, unsafe_allow_javascript=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("→ Enter Hospital Intelligence System", use_container_width=True, type="primary"):
            st.session_state.landing_seen = True
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════

if not st.session_state.landing_seen:
    page_landing()
elif not st.session_state.logged_in:
    page_login()
elif st.session_state.role == "patient":
    patient_dashboard()
elif st.session_state.role == "admin":
    admin_dashboard()
else:
    st.error("Unknown role. Please log out and try again.")
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()
