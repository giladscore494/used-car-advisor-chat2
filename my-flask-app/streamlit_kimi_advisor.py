# -*- coding: utf-8 -*-
"""
ניסוי Kimi K2.6 – המלצות רכב ישראלי
Streamlit test app for car recommendations using Kimi K2.6 with web search.
"""

import streamlit as st
import json
import os
import sys
import openai
import httpx

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="ניסוי Kimi K2.6 – המלצות רכב", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { direction: rtl; text-align: right; }
    .car-card {
        background: #fff; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.08);
        padding: 16px; margin-bottom: 14px; border-right: 4px solid #2259b4;
    }
    .disclaimer { color:#a16207; background:#fffbeb; border:1px solid #fde68a;
                   padding:8px 12px; border-radius:10px; margin-bottom:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Kimi client
# --------------------------------------------------
KIMI_MODEL = "kimi-k2.6"
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
MAX_TOKENS = 12000
MAX_TOOL_CALL_ITERATIONS = 10
KIMI_TOOLS = [{"type": "builtin_function", "function": {"name": "$web_search"}}]


def get_kimi_client():
    api_key = None

    try:
        api_key = st.secrets.get("MOONSHOT_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("MOONSHOT_API_KEY")

    if not api_key:
        return None, "Missing MOONSHOT_API_KEY. Add it to Streamlit secrets or environment variables."

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=KIMI_BASE_URL,
        )
        return client, None
    except TypeError as exc:
        return None, (
            "Failed to create OpenAI-compatible client. "
            "This usually means an openai/httpx dependency mismatch. "
            "Check requirements.txt pins: openai>=1.55.3,<2 and httpx==0.27.2. "
            f"Original error: {exc}"
        )
    except Exception as exc:
        return None, f"Failed to create Kimi client: {type(exc).__name__}: {exc}"


# --------------------------------------------------
# Prompt builder
# --------------------------------------------------
SYSTEM_PROMPT = """You are an independent automotive data analyst for the Israeli used-car market.
Recommend cars for the user profile below.
You must use internet search to verify Israeli-market reality.

Critical rules:
- Focus only on Israeli used cars.
- Do not invent models, trims, prices, safety ratings, warranty data, license fees, common faults, or market supply.
- If a field is not verified from a reliable source, return null or "unknown".
- Return only one valid JSON object.
- No markdown.
- No text before or after JSON.
- Fit score means preference fit only, not purchase approval.
- Do not use first-person language like "אני ממליץ", "הייתי קונה", "תקנה", or "אל תקנה".
- Always separate preference fit from risks.
- Use Hebrew for user-facing explanation fields.

Required output schema:
{
  "search_performed": true,
  "search_queries": [],
  "recommended_cars": [
    {
      "brand": "",
      "model": "",
      "year_range": "",
      "fuel": "",
      "gear": "",
      "turbo": null,
      "engine_cc": null,
      "price_range_nis": [null, null],
      "avg_fuel_consumption": null,
      "fuel_method": "",
      "annual_fee": null,
      "fee_method": "official|unknown",
      "reliability_score": null,
      "reliability_method": "",
      "maintenance_cost": null,
      "maintenance_method": "",
      "safety_rating": null,
      "safety_method": "",
      "insurance_cost": null,
      "insurance_method": "",
      "resale_value": null,
      "resale_method": "",
      "performance_score": null,
      "performance_method": "",
      "comfort_features": null,
      "comfort_method": "",
      "suitability": null,
      "suitability_method": "",
      "market_supply": "גבוה|בינוני|נמוך|unknown",
      "supply_method": "",
      "fit_score": null,
      "comparison_comment": "",
      "not_recommended_reason": "",
      "best_for": [],
      "not_ideal_for": [],
      "practical_summary": "",
      "sources": []
    }
  ],
  "general_notes": [],
  "limitations": []
}"""


def build_user_message(profile: dict) -> str:
    return f"User profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}"


# --------------------------------------------------
# Helper functions for handling SDK objects and dicts
# --------------------------------------------------
def obj_get(obj, key, default=None):
    """Safely get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_message(message):
    """Convert message to dict format."""
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return dict(message)


def get_tool_call_id(tool_call):
    """Extract tool call id from object or dict."""
    return obj_get(tool_call, "id")


def get_tool_call_function(tool_call):
    """Extract function from tool call object or dict."""
    return obj_get(tool_call, "function", {})


def get_tool_call_name(tool_call):
    """Extract function name from tool call."""
    function = get_tool_call_function(tool_call)
    return obj_get(function, "name")


def get_tool_call_arguments(tool_call):
    """Extract function arguments from tool call."""
    function = get_tool_call_function(tool_call)
    return obj_get(function, "arguments", "{}")


# --------------------------------------------------
# Kimi tool-call loop
# --------------------------------------------------
def call_kimi(client, profile: dict) -> dict:
    """Send request to Kimi K2.6 with web_search, handle tool-call loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(profile)},
    ]

    finish_reason = None

    for _ in range(MAX_TOOL_CALL_ITERATIONS):  # prevent infinite loops
        completion = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            tools=KIMI_TOOLS,
            temperature=1.0,
            max_tokens=MAX_TOKENS,
            extra_body={
                "thinking": {"type": "disabled"}
            },
        )

        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        assistant_message = normalize_message(choice.message)

        if finish_reason == "tool_calls":
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []

            for tool_call in tool_calls:
                tool_call_id = get_tool_call_id(tool_call)
                tool_name = get_tool_call_name(tool_call)
                tool_args = get_tool_call_arguments(tool_call)

                if not tool_call_id:
                    raise RuntimeError("Kimi tool call missing id")

                if not tool_name:
                    raise RuntimeError("Kimi tool call missing function.name")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_args or "{}",
                })

            continue

        if finish_reason == "stop":
            final_content = assistant_message.get("content", "")
            return {"raw": final_content, "usage": completion.usage}

        raise RuntimeError(f"Unexpected Kimi finish_reason: {finish_reason}")

    return {"raw": "Error: tool-call loop exceeded max iterations.", "usage": None}


def parse_kimi_result(raw: str) -> dict | None:
    """Try to parse the raw Kimi response as JSON."""
    if not raw:
        return None
    # Strip possible markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------
# Session state helpers
# --------------------------------------------------
def init_state():
    if "kimi_result" not in st.session_state:
        st.session_state.kimi_result = None
    if "kimi_raw" not in st.session_state:
        st.session_state.kimi_raw = None


init_state()

# --------------------------------------------------
# UI – Title & Disclaimer
# --------------------------------------------------
st.title("🚗 ניסוי Kimi K2.6 – המלצות רכב")
st.markdown(
    '<div class="disclaimer">⚠️ זה ניסוי בלבד. לא המלצת קנייה סופית. '
    "יש לבדוק כל רכב במכון ובמקורות רשמיים.</div>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("Debug"):
    st.write("Python:", sys.version)
    st.write("openai:", getattr(openai, "__version__", "unknown"))
    st.write("httpx:", getattr(httpx, "__version__", "unknown"))

# --------------------------------------------------
# Questionnaire
# --------------------------------------------------
st.subheader("📋 שאלון המלצות")

col1, col2 = st.columns(2)

with col1:
    budget_min = st.number_input("תקציב מינימלי (₪)", min_value=0, value=40000, step=5000, key="budget_min")
    year_min = st.number_input("שנת ייצור מינימלית", min_value=2000, max_value=2026, value=2015, key="year_min")
    fuels_he = st.multiselect("סוג דלק", ["בנזין", "דיזל", "היברידי", "חשמלי"], default=["בנזין"], key="fuels_he")
    turbo_choice_he = st.radio("טורבו", ["לא משנה", "כן", "לא"], horizontal=True, key="turbo_he")
    seats_choice = st.selectbox("מספר מושבים", ["5", "7", "8+"], key="seats")
    annual_km = st.number_input("ק\"מ שנתי", min_value=1, value=15000, step=1000, key="annual_km")
    family_size = st.selectbox("גודל משפחה", ["1-2", "3-4", "5+"], key="family_size")
    driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח", "משולב", "דינמי"], key="driving_style")

with col2:
    budget_max = st.number_input("תקציב מקסימלי (₪)", min_value=0, value=120000, step=5000, key="budget_max")
    year_max = st.number_input("שנת ייצור מקסימלית", min_value=2000, max_value=2026, value=2024, key="year_max")
    gears_he = st.multiselect("סוג תיבת הילוכים", ["אוטומטית", "ידנית"], default=["אוטומטית"], key="gears_he")
    body_style = st.selectbox("סוג מרכב", ["כללי", "משפחתי", "ג'יפון", "סדאן", "האצ׳בק", "מסחרי"], key="body_style")
    trim_level = st.selectbox("רמת גימור", ["סטנדרטי", "מפנק", "מינימלי"], key="trim_level")
    driver_age = st.number_input("גיל נהג", min_value=17, max_value=120, value=30, key="driver_age")
    cargo_need = st.selectbox("צורך במטען", ["קטן", "בינוני", "גדול"], key="cargo_need")
    safety_required = st.radio("בטיחות נדרשת", ["כן", "לא"], horizontal=True, key="safety_req")

main_use = st.text_input("שימוש עיקרי (למשל: נסיעה לעבודה, טיולי שטח)", value="נסיעה יומית לעבודה", key="main_use")
consider_supply = st.radio("להתחשב בהיצע בשוק?", ["כן", "לא"], horizontal=True, key="consider_supply")

# Weights
st.markdown("**משקולות חשיבות (1-10):**")
w1, w2, w3, w4, w5 = st.columns(5)
with w1:
    w_reliability = st.slider("אמינות", 1, 10, 8, key="w_rel")
with w2:
    w_resale = st.slider("שמירת ערך", 1, 10, 6, key="w_res")
with w3:
    w_fuel = st.slider("חיסכון בדלק", 1, 10, 7, key="w_fuel")
with w4:
    w_performance = st.slider("ביצועים", 1, 10, 5, key="w_perf")
with w5:
    w_comfort = st.slider("נוחות", 1, 10, 6, key="w_comf")

# Advanced fields in expander
with st.expander("⚙️ שדות מתקדמים"):
    excluded_colors = st.text_input("צבעים לא רצויים (מופרדים בפסיק)", value="", key="excluded_colors")
    fuel_price = st.number_input("מחיר דלק (₪/ליטר)", min_value=0.0, value=7.0, step=0.1, key="fuel_price")
    electricity_price = st.number_input("מחיר חשמל (₪/קוט\"ש)", min_value=0.0, value=0.55, step=0.05, key="elec_price")
    license_years = st.number_input("שנות רישיון", min_value=0, max_value=70, value=5, key="license_years")

# --------------------------------------------------
# Validation
# --------------------------------------------------
errors: list[str] = []
if budget_max <= budget_min:
    errors.append("תקציב מקסימלי חייב להיות גדול מהמינימלי.")
if year_max < year_min:
    errors.append("שנת ייצור מקסימלית חייבת להיות גדולה או שווה למינימלית.")
if not fuels_he:
    errors.append("יש לבחור לפחות סוג דלק אחד.")
if not gears_he and "חשמלי" not in fuels_he:
    errors.append("יש לבחור לפחות סוג תיבת הילוכים אחד (אלא אם נבחר חשמלי).")
if driver_age < 17:
    errors.append("גיל נהג חייב להיות 17 ומעלה.")
if annual_km <= 0:
    errors.append("ק\"מ שנתי חייב להיות חיובי.")

for err in errors:
    st.error(err)

# --------------------------------------------------
# Safety controls
# --------------------------------------------------
st.markdown("---")
cost_ack = st.checkbox("אני מבין שזה ניסוי עם עלות API", key="cost_ack")

col_btn, col_clear = st.columns([1, 1])
with col_btn:
    submit_disabled = bool(errors) or not cost_ack
    submit_clicked = st.button("🔍 קבל המלצות עם Kimi", disabled=submit_disabled, type="primary")
with col_clear:
    if st.button("🗑️ נקה תוצאות"):
        st.session_state.kimi_result = None
        st.session_state.kimi_raw = None
        st.rerun()

# --------------------------------------------------
# Build profile & call Kimi
# --------------------------------------------------
turbo_map = {"לא משנה": None, "כן": True, "לא": False}
excluded = [c.strip() for c in excluded_colors.split(",") if c.strip()] if excluded_colors else []

if submit_clicked and not errors:
    user_profile = {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(year_min), int(year_max)],
        "fuel": fuels_he,
        "gear": gears_he,
        "turbo_required": turbo_map[turbo_choice_he],
        "main_use": main_use,
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
        "license_years": int(license_years),
        "family_size": family_size,
        "cargo_need": cargo_need,
        "safety_required": safety_required,
        "trim_level": trim_level,
        "weights": {
            "reliability": w_reliability,
            "resale": w_resale,
            "fuel": w_fuel,
            "performance": w_performance,
            "comfort": w_comfort,
        },
        "body_style": body_style,
        "driving_style": driving_style,
        "excluded_colors": excluded,
        "consider_market_supply": consider_supply == "כן",
        "fuel_price_nis_per_liter": float(fuel_price),
        "electricity_price_nis_per_kwh": float(electricity_price),
        "seats": seats_choice,
    }

    client, err = get_kimi_client()
    if err:
        st.error(err)
        st.stop()

    with st.spinner("🔄 Kimi K2.6 מחפש ומנתח... (עשוי לקחת עד דקה)"):
        try:
            result = call_kimi(client, user_profile)
            st.session_state.kimi_raw = result["raw"]
            parsed = parse_kimi_result(result["raw"])
            if parsed:
                st.session_state.kimi_result = parsed
            else:
                st.session_state.kimi_result = None
                st.error("❌ לא הצלחתי לפרסר את תשובת Kimi כ-JSON.")
                with st.expander("תשובה גולמית"):
                    st.code(result["raw"])
        except RuntimeError as exc:
            st.error(f"❌ שגיאה ב-Kimi: {exc}")
            st.session_state.kimi_result = None
            st.session_state.kimi_raw = None
        except Exception as exc:
            st.error(f"❌ שגיאה בלתי צפויה בקריאה ל-Kimi: {type(exc).__name__}: {exc}")
            st.session_state.kimi_result = None
            st.session_state.kimi_raw = None

# --------------------------------------------------
# Display results
# --------------------------------------------------
if st.session_state.kimi_result:
    data = st.session_state.kimi_result
    cars = data.get("recommended_cars", [])

    st.subheader(f"🏆 נמצאו {len(cars)} המלצות")

    for car in cars:
        brand = car.get("brand", "")
        model = car.get("model", "")
        year_range = car.get("year_range", "")
        price = car.get("price_range_nis", [None, None])
        price_str = f"₪{price[0]:,.0f} – ₪{price[1]:,.0f}" if price and price[0] and price[1] else "לא ידוע"
        fit = car.get("fit_score")
        fit_str = f"⭐ {fit}/10" if fit is not None else ""

        st.markdown(
            f'<div class="car-card">'
            f"<h3>{brand} {model} ({year_range}) {fit_str}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("טווח מחיר", price_str)
            st.metric("דלק", car.get("fuel", "—"))
            st.metric("תיבה", car.get("gear", "—"))
            st.metric("נפח מנוע", car.get("engine_cc") or "—")
        with c2:
            st.metric("אמינות", f'{car.get("reliability_score", "—")}/10')
            st.metric("עלות אחזקה שנתית", f'₪{car.get("maintenance_cost", "—")}')
            st.metric("היצע בשוק", car.get("market_supply", "—"))
        with c3:
            st.metric("בטיחות", f'{car.get("safety_rating", "—")}/10')
            st.metric("שמירת ערך", f'{car.get("resale_value", "—")}/10')
            st.metric("ביצועים", f'{car.get("performance_score", "—")}/10')

        comment = car.get("comparison_comment", "")
        if comment:
            st.info(f"💬 {comment}")

        not_rec = car.get("not_recommended_reason", "")
        if not_rec:
            st.warning(f"⚠️ {not_rec}")

        practical = car.get("practical_summary", "")
        if practical:
            st.caption(practical)

        sources = car.get("sources", [])
        if sources:
            with st.expander("מקורות"):
                for src in sources:
                    st.write(f"- {src}")

        st.markdown("---")

    # General notes
    notes = data.get("general_notes", [])
    if notes:
        with st.expander("📝 הערות כלליות"):
            for note in notes:
                st.write(f"• {note}")

    limitations = data.get("limitations", [])
    if limitations:
        with st.expander("⚠️ מגבלות"):
            for lim in limitations:
                st.write(f"• {lim}")

    # Search queries
    queries = data.get("search_queries", [])
    if queries:
        with st.expander("🔎 שאילתות חיפוש שבוצעו"):
            for q in queries:
                st.write(f"• {q}")

    # Raw JSON
    with st.expander("📄 JSON גולמי"):
        st.json(data)

elif st.session_state.kimi_raw and not st.session_state.kimi_result:
    # JSON parse failed but we have raw content
    with st.expander("תשובה גולמית (לא JSON תקין)"):
        st.code(st.session_state.kimi_raw)
