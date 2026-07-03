"""
Threshold-based decision engine.
All functions return a dict: {status, color, message, emoji}.
Colors are CSS hex strings compatible with st.markdown.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def occupancy_alert(rate: float) -> dict:
    """
    WHO 85 % bed-occupancy threshold.
    rate: float 0–1 (or 0–100 if >1 is passed — normalised internally).
    """
    if rate > 1.0:
        rate = rate / 100.0
    if rate > config.WHO_OCCUPANCY_THRESHOLD:
        return dict(
            status  = "SURGE",
            color   = "#C0392B",
            bg      = "#FDEDEC",
            message = "Surge predicted — expand capacity immediately.",
            emoji   = "🔴",
        )
    if rate > config.WARN_OCCUPANCY_THRESHOLD:
        return dict(
            status  = "WARNING",
            color   = "#D68910",
            bg      = "#FEF9E7",
            message = "Approaching capacity — monitor closely.",
            emoji   = "🟡",
        )
    return dict(
        status  = "NORMAL",
        color   = "#1D6A39",
        bg      = "#EAFAF1",
        message = "Occupancy within safe limits.",
        emoji   = "🟢",
    )


def overtime_alert(rate: float) -> dict:
    """
    rate: fraction of shifts that are overtime (0–1).
    Threshold: >20 % = breach.
    """
    if rate > 1.0:
        rate = rate / 100.0
    if rate > config.OVERTIME_THRESHOLD:
        return dict(
            status  = "BREACH",
            color   = "#C0392B",
            bg      = "#FDEDEC",
            message = f"Overtime breach: {rate*100:.1f}% of shifts (limit 20%). Review staffing.",
            emoji   = "🔴",
        )
    return dict(
        status  = "OK",
        color   = "#1D6A39",
        bg      = "#EAFAF1",
        message = f"Overtime within limit: {rate*100:.1f}% of shifts.",
        emoji   = "🟢",
    )


def los_flag(days: float) -> dict:
    """
    Flag a length-of-stay value against the WHO 7-day benchmark.
    """
    if days > config.WHO_LOS_DAYS * 1.5:
        return dict(
            status  = "EXTENDED",
            color   = "#C0392B",
            bg      = "#FDEDEC",
            message = f"LOS {days:.1f} d — significantly above WHO 7-day benchmark.",
            emoji   = "🔴",
        )
    if days > config.WHO_LOS_DAYS:
        return dict(
            status  = "ABOVE BENCHMARK",
            color   = "#D68910",
            bg      = "#FEF9E7",
            message = f"LOS {days:.1f} d — above WHO 7-day benchmark.",
            emoji   = "🟡",
        )
    return dict(
        status  = "WITHIN BENCHMARK",
        color   = "#1D6A39",
        bg      = "#EAFAF1",
        message = f"LOS {days:.1f} d — within WHO 7-day benchmark.",
        emoji   = "🟢",
    )


def surge_recommendation(rate: float, extra_admissions_pct: float = 0) -> dict:
    """
    Combined occupancy + simulated load recommendation for admin Surge tab.
    extra_admissions_pct: 0–100 slider value representing % increase in admissions.
    """
    simulated_rate = min(1.0, rate * (1 + extra_admissions_pct / 100))
    base = occupancy_alert(simulated_rate)

    beds_shortage = 0
    if simulated_rate > config.WHO_OCCUPANCY_THRESHOLD:
        # Estimate number of extra beds needed to bring back to 80 %
        safe_target   = 0.80
        total_beds_est = 100  # normalised; admin tab uses real counts
        beds_shortage  = round((simulated_rate - safe_target) * total_beds_est)

    base["simulated_rate"] = simulated_rate
    base["beds_shortage"]  = beds_shortage
    return base
