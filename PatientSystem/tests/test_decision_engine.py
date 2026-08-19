"""Unit tests for src/decision_engine.py — pure threshold logic, no I/O."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decision_engine import (
    occupancy_alert,
    overtime_alert,
    los_flag,
    surge_recommendation,
)


# ── occupancy_alert ──────────────────────────────────────────────────────────

def test_occupancy_below_warning_is_normal():
    assert occupancy_alert(0.50)["status"] == "NORMAL"


def test_occupancy_at_warning_threshold_is_still_normal():
    # threshold is strictly ">", so the boundary value itself must not trip it
    assert occupancy_alert(0.75)["status"] == "NORMAL"


def test_occupancy_just_above_warning_threshold_is_warning():
    assert occupancy_alert(0.751)["status"] == "WARNING"


def test_occupancy_at_surge_threshold_is_still_warning():
    assert occupancy_alert(0.85)["status"] == "WARNING"


def test_occupancy_just_above_surge_threshold_is_surge():
    assert occupancy_alert(0.851)["status"] == "SURGE"


def test_occupancy_accepts_0_to_100_scale_and_normalises():
    # 90 (percent-style) must behave identically to 0.90 (fraction-style)
    assert occupancy_alert(90)["status"] == occupancy_alert(0.90)["status"] == "SURGE"


def test_occupancy_alert_shape():
    result = occupancy_alert(0.5)
    assert set(result.keys()) >= {"status", "color", "bg", "message", "emoji"}


# ── overtime_alert ───────────────────────────────────────────────────────────

def test_overtime_at_threshold_is_ok():
    assert overtime_alert(0.20)["status"] == "OK"


def test_overtime_just_above_threshold_is_breach():
    assert overtime_alert(0.201)["status"] == "BREACH"


def test_overtime_accepts_0_to_100_scale():
    assert overtime_alert(25)["status"] == overtime_alert(0.25)["status"] == "BREACH"


def test_overtime_breach_message_reports_percentage():
    msg = overtime_alert(0.334)["message"]
    assert "33.4%" in msg


# ── los_flag ──────────────────────────────────────────────────────────────────

def test_los_at_benchmark_is_within():
    assert los_flag(7.0)["status"] == "WITHIN BENCHMARK"


def test_los_just_above_benchmark_is_above():
    assert los_flag(7.01)["status"] == "ABOVE BENCHMARK"


def test_los_at_extended_threshold_is_still_above_not_extended():
    # WHO_LOS_DAYS * 1.5 == 10.5; the boundary itself must not tip into EXTENDED
    assert los_flag(10.5)["status"] == "ABOVE BENCHMARK"


def test_los_just_above_extended_threshold_is_extended():
    assert los_flag(10.51)["status"] == "EXTENDED"


def test_los_well_below_benchmark_is_within():
    assert los_flag(2.0)["status"] == "WITHIN BENCHMARK"


# ── surge_recommendation ───────────────────────────────────────────────────────

def test_surge_recommendation_no_extra_load_matches_plain_occupancy_alert():
    result = surge_recommendation(0.90, extra_admissions_pct=0)
    assert result["status"] == "SURGE"
    assert result["simulated_rate"] == 0.90


def test_surge_recommendation_simulated_rate_is_capped_at_one():
    # 0.90 * (1 + 50/100) = 1.35, which must be capped to 1.0
    result = surge_recommendation(0.90, extra_admissions_pct=50)
    assert result["simulated_rate"] == 1.0


def test_surge_recommendation_beds_shortage_only_computed_when_surging():
    calm = surge_recommendation(0.50, extra_admissions_pct=0)
    assert calm["status"] == "NORMAL"
    assert calm["beds_shortage"] == 0

    surging = surge_recommendation(0.90, extra_admissions_pct=0)
    assert surging["status"] == "SURGE"
    # (0.90 - 0.80) * 100 == 10, per the function's normalised-100-bed estimate
    assert surging["beds_shortage"] == 10


def test_surge_recommendation_extra_load_can_push_normal_into_surge():
    baseline = occupancy_alert(0.70)["status"]
    assert baseline == "NORMAL"
    pushed = surge_recommendation(0.70, extra_admissions_pct=30)
    assert pushed["simulated_rate"] == pytest.approx(0.91)
    assert pushed["status"] == "SURGE"
