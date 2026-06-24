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
SYSTEM_PROMPT = """You are an Israeli used-car market analyst.
Recommend cars for the user profile below.
You MUST use web search to verify Israeli-market reality.
You MUST prefer Israeli-market sources and official/structured sources.

Critical rules:
- Focus ONLY on Israeli used cars.
- Do NOT invent ratings, prices, fees, trims, faults, or market supply.
- Use official ratings ONLY when there is an official source (e.g., Euro NCAP, Israeli safety grade).
- If no official source exists, return "not_official" or null and explain the estimate basis.
- Do NOT create arbitrary 1-10 scores for reliability, safety, resale, comfort, insurance, or maintenance.
- Safety should be based on official crash-test source when available, preferably Euro NCAP for the correct generation/year range.
- Fuel consumption should separate: official_consumption, real_world_estimate, reasoning.
- Annual license fee should use official Israeli calculation/source when possible. If not verified, return null.
- Maintenance cost must include a reasoning breakdown and common faults by component.
- Common faults must be tied to age, mileage, engine, gearbox, fuel type, turbo/hybrid/electric system, and known technical components.
- Market supply must be based on visible Israeli listing/search evidence when possible. If not verified, return "unknown".
- Every recommended car must include sources.
- If sources are missing, the car must be marked as "needs_review" and not as "verified".
- Return only valid JSON. No markdown. No text before or after JSON.
- Use Hebrew for user-facing explanation fields.

Required output schema:
{
  "search_performed": true,
  "search_queries": [],
  "recommended_cars": [
    {
      "brand": "",
      "model": "",
      "generation_or_year_range": "",
      "recommended_years": "",
      "fuel": "",
      "gear": "",
      "engine_or_drivetrain": "",
      "turbo": null,
      "body_style": "",
      "seats": "",
      
      "price_analysis": {
        "estimated_price_range_nis": [null, null],
        "confidence": "high|medium|low|unknown",
        "basis": "",
        "sources": []
      },
      
      "official_ratings": {
        "safety": {
          "rating": null,
          "rating_system": "Euro NCAP|IIHS|Israeli safety grade|unknown",
          "year_tested": null,
          "generation_match": "exact|close|uncertain|unknown",
          "source": ""
        },
        "emissions_or_green_score": {
          "value": null,
          "system": "Israeli green score|Euro standard|unknown",
          "source": ""
        },
        "official_fuel_consumption": {
          "value": null,
          "unit": "km/l|l/100km|kWh/100km|unknown",
          "source": ""
        },
        "annual_license_fee": {
          "estimated_nis": null,
          "method": "official|estimated|unknown",
          "reasoning": "",
          "source": ""
        }
      },
      
      "real_world_use": {
        "real_world_fuel_estimate": null,
        "fuel_estimate_reasoning": "",
        "city_vs_highway_note": "",
        "comfort_practical_note": "",
        "performance_practical_note": ""
      },
      
      "maintenance_analysis": {
        "estimated_annual_maintenance_range_nis": [null, null],
        "confidence": "high|medium|low|unknown",
        "calculation_reasoning": "",
        "assumptions": {
          "vehicle_age_years": null,
          "annual_km": null,
          "likely_mileage_range": "",
          "service_history_importance": ""
        },
        "common_faults_by_component": [
          {
            "component": "engine|gearbox|turbo|hybrid_system|battery|cooling|suspension|brakes|electronics|ac|body|other",
            "fault": "",
            "risk_level": "low|medium|high|unknown",
            "more_likely_when": "high mileage|poor maintenance|city driving|age|known generation issue|unknown",
            "inspection_advice": "",
            "source": ""
          }
        ],
        "expensive_risk_items": [],
        "cheap_common_items": []
      },
      
      "ownership_risk": {
        "main_risks": [],
        "what_to_check_before_buying": [],
        "avoid_if": [],
        "good_candidate_if": []
      },
      
      "market_supply": {
        "level": "high|medium|low|unknown",
        "basis": "",
        "sources": []
      },
      
      "fit_analysis": {
        "why_it_fits": "",
        "why_it_may_not_fit": "",
        "best_for": [],
        "not_ideal_for": [],
        "confidence": "high|medium|low|unknown"
      },
      
      "recommendation_status": "verified|needs_review|not_recommended",
      "sources": []
    }
  ],
  "rejected_options": [
    {
      "brand": "",
      "model": "",
      "reason": "",
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
            max_tokens=MAX_TOKENS,
            extra_body={"thinking": {"type": "disabled"}},
        )

        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        assistant_message = normalize_message(choice.message)

        if finish_reason == "tool_calls":
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])

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
# Source validation and local fit classification
# --------------------------------------------------
def validate_sources(car: dict) -> dict:
    """Validate that car has sufficient sources."""
    sources = car.get("sources", [])
    
    # Count sources by type
    has_price_source = False
    has_technical_source = False
    
    for src in sources:
        if isinstance(src, dict):
            supports = src.get("supports", "")
            if any(x in supports for x in ["price", "market"]):
                has_price_source = True
            if any(x in supports for x in ["safety", "fuel", "maintenance", "faults", "official"]):
                has_technical_source = True
        elif isinstance(src, str):
            # Legacy string source - count as general
            has_technical_source = True
    
    return {
        "has_sources": len(sources) > 0,
        "has_price_source": has_price_source,
        "has_technical_source": has_technical_source,
        "source_count": len(sources)
    }


def calculate_local_fit(car: dict, profile: dict) -> dict:
    """Calculate local fit classification based on evidence."""
    # Get recommendation status
    status = car.get("recommendation_status", "needs_review")
    
    if status == "not_recommended":
        return {
            "level": "low",
            "label": "התאמה נמוכה",
            "color": "#dc2626"
        }
    
    # Validate sources
    source_validation = validate_sources(car)
    
    # Check price match
    price_analysis = car.get("price_analysis", {})
    price_range = price_analysis.get("estimated_price_range_nis", [None, None])
    budget_min = profile.get("budget_nis", [0, 0])[0]
    budget_max = profile.get("budget_nis", [0, 999999])[1]
    
    price_in_budget = False
    if price_range[0] and price_range[1]:
        # Check if there's overlap between price range and budget
        price_in_budget = price_range[0] <= budget_max and price_range[1] >= budget_min
    
    # Check maintenance risk
    maintenance = car.get("maintenance_analysis", {})
    high_risk_faults = sum(
        1 for fault in maintenance.get("common_faults_by_component", [])
        if fault.get("risk_level") == "high"
    )
    
    # Calculate fit level
    fit_score = 0
    
    # Status weight (most important)
    if status == "verified":
        fit_score += 40
    elif status == "needs_review":
        fit_score += 20
    
    # Source quality
    if source_validation["has_sources"]:
        fit_score += 15
    if source_validation["has_price_source"]:
        fit_score += 10
    if source_validation["has_technical_source"]:
        fit_score += 10
    
    # Price match
    if price_in_budget:
        fit_score += 15
    
    # Risk level
    if high_risk_faults == 0:
        fit_score += 10
    elif high_risk_faults <= 2:
        fit_score += 5
    
    # Classify
    if fit_score >= 75:
        return {
            "level": "high",
            "label": "התאמה גבוהה",
            "color": "#16a34a"
        }
    elif fit_score >= 50:
        return {
            "level": "medium",
            "label": "התאמה בינונית",
            "color": "#ca8a04"
        }
    elif fit_score >= 30:
        return {
            "level": "review",
            "label": "דורש בדיקה",
            "color": "#ea580c"
        }
    else:
        return {
            "level": "low",
            "label": "התאמה נמוכה",
            "color": "#dc2626"
        }


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
    rejected = data.get("rejected_options", [])
    
    # Build user profile for fit calculation
    user_profile = {
        "budget_nis": [float(budget_min), float(budget_max)],
    }

    # Separate cars by status
    verified_cars = []
    review_cars = []
    not_recommended_cars = []
    
    for car in cars:
        status = car.get("recommendation_status", "needs_review")
        if status == "verified":
            verified_cars.append(car)
        elif status == "needs_review":
            review_cars.append(car)
        else:
            not_recommended_cars.append(car)

    # Display verified recommendations
    if verified_cars:
        st.subheader(f"✅ המלצות מאומתות ({len(verified_cars)})")
        for car in verified_cars:
            display_car_card(car, user_profile)

    # Display needs review
    if review_cars:
        st.subheader(f"🔍 אפשרויות לבדיקה נוספת ({len(review_cars)})")
        for car in review_cars:
            display_car_card(car, user_profile)

    # Display not recommended
    if not_recommended_cars:
        st.subheader(f"⛔ נפסלו / לא מתאימים ({len(not_recommended_cars)})")
        for car in not_recommended_cars:
            display_car_card(car, user_profile, show_minimal=True)
    
    # Display rejected options
    if rejected:
        st.subheader(f"❌ אפשרויות שנפסלו ({len(rejected)})")
        for rej in rejected:
            brand = rej.get("brand", "")
            model = rej.get("model", "")
            reason = rej.get("reason", "")
            st.markdown(f"**{brand} {model}**: {reason}")

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


def display_car_card(car: dict, profile: dict, show_minimal: bool = False):
    """Display a car card with the new schema."""
    brand = car.get("brand", "")
    model = car.get("model", "")
    generation = car.get("generation_or_year_range", "")
    recommended_years = car.get("recommended_years", "")
    
    # Calculate local fit
    fit = calculate_local_fit(car, profile)
    fit_label = fit["label"]
    fit_color = fit["color"]
    
    # Build header
    header = f"{brand} {model}"
    if generation:
        header += f" ({generation})"
    if recommended_years:
        header += f" | מומלץ: {recommended_years}"
    
    st.markdown(
        f'<div class="car-card">'
        f'<h3>{header}</h3>'
        f'<p style="color:{fit_color}; font-weight:bold;">רמת התאמה משוערת: {fit_label}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
    
    if show_minimal:
        # For not recommended, show minimal info
        st.caption(f"דלק: {car.get('fuel', '—')} | תיבה: {car.get('gear', '—')}")
        return
    
    # Basic info
    st.caption(
        f"🚗 {car.get('body_style', '—')} | "
        f"⛽ {car.get('fuel', '—')} | "
        f"⚙️ {car.get('gear', '—')} | "
        f"🔧 {car.get('engine_or_drivetrain', '—')} | "
        f"💺 {car.get('seats', '—')} מושבים"
    )
    
    # Price analysis
    price_analysis = car.get("price_analysis", {})
    price_range = price_analysis.get("estimated_price_range_nis", [None, None])
    price_str = f"₪{price_range[0]:,.0f} – ₪{price_range[1]:,.0f}" if price_range[0] and price_range[1] else "לא ידוע"
    price_confidence = price_analysis.get("confidence", "unknown")
    price_basis = price_analysis.get("basis", "")
    
    st.markdown(f"**💰 מחיר משוער**: {price_str} (רמת ודאות: {price_confidence})")
    if price_basis:
        st.caption(f"בסיס: {price_basis}")
    
    # Official ratings
    official = car.get("official_ratings", {})
    
    # Safety rating
    safety = official.get("safety", {})
    safety_rating = safety.get("rating")
    if safety_rating:
        safety_system = safety.get("rating_system", "")
        year_tested = safety.get("year_tested", "")
        generation_match = safety.get("generation_match", "")
        safety_source = safety.get("source", "")
        st.markdown(
            f"**🛡️ בטיחות רשמית**: {safety_rating} ({safety_system})"
        )
        st.caption(
            f"שנת בדיקה: {year_tested} | התאמת דור: {generation_match}"
        )
        if safety_source:
            st.caption(f"מקור: {safety_source}")
    else:
        st.markdown("**🛡️ בטיחות רשמית**: לא אומת")
    
    # Official fuel consumption
    official_fuel = official.get("official_fuel_consumption", {})
    fuel_value = official_fuel.get("value")
    if fuel_value:
        fuel_unit = official_fuel.get("unit", "")
        fuel_source = official_fuel.get("source", "")
        st.markdown(f"**⛽ צריכת דלק רשמית**: {fuel_value} {fuel_unit}")
        if fuel_source:
            st.caption(f"מקור: {fuel_source}")
    
    # Real world fuel estimate
    real_world = car.get("real_world_use", {})
    real_fuel = real_world.get("real_world_fuel_estimate")
    if real_fuel:
        fuel_reasoning = real_world.get("fuel_estimate_reasoning", "")
        st.markdown(f"**⛽ צריכת דלק בפועל (משוערת)**: {real_fuel}")
        if fuel_reasoning:
            st.caption(f"הסבר: {fuel_reasoning}")
    
    # Annual license fee
    license_fee = official.get("annual_license_fee", {})
    fee_nis = license_fee.get("estimated_nis")
    if fee_nis:
        fee_method = license_fee.get("method", "")
        fee_reasoning = license_fee.get("reasoning", "")
        st.markdown(f"**💳 אגרת רישוי שנתית**: ₪{fee_nis:,.0f} ({fee_method})")
        if fee_reasoning:
            st.caption(f"הסבר: {fee_reasoning}")
    
    # Maintenance analysis
    maintenance = car.get("maintenance_analysis", {})
    maint_range = maintenance.get("estimated_annual_maintenance_range_nis", [None, None])
    if maint_range[0] and maint_range[1]:
        maint_confidence = maintenance.get("confidence", "unknown")
        maint_reasoning = maintenance.get("calculation_reasoning", "")
        st.markdown(
            f"**🔧 עלות אחזקה שנתית משוערת**: "
            f"₪{maint_range[0]:,.0f} – ₪{maint_range[1]:,.0f} "
            f"(ודאות: {maint_confidence})"
        )
        if maint_reasoning:
            st.caption(f"בסיס חישוב: {maint_reasoning}")
    
    # Common faults
    faults = maintenance.get("common_faults_by_component", [])
    if faults:
        with st.expander("⚠️ תקלות נפוצות לפי רכיב"):
            for fault in faults:
                component = fault.get("component", "")
                fault_desc = fault.get("fault", "")
                risk = fault.get("risk_level", "unknown")
                when = fault.get("more_likely_when", "")
                advice = fault.get("inspection_advice", "")
                
                risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
                st.markdown(f"**{risk_icon} {component}**: {fault_desc}")
                if when:
                    st.caption(f"סביר יותר כאשר: {when}")
                if advice:
                    st.caption(f"עצה לבדיקה: {advice}")
    
    expensive_risks = maintenance.get("expensive_risk_items", [])
    if expensive_risks:
        st.warning(f"**פריטי סיכון יקרים**: {', '.join(expensive_risks)}")
    
    cheap_common = maintenance.get("cheap_common_items", [])
    if cheap_common:
        st.info(f"**פריטים נפוצים זולים**: {', '.join(cheap_common)}")
    
    # Ownership risk
    ownership = car.get("ownership_risk", {})
    main_risks = ownership.get("main_risks", [])
    if main_risks:
        with st.expander("⚡ סיכונים עיקריים"):
            for risk in main_risks:
                st.write(f"• {risk}")
    
    what_to_check = ownership.get("what_to_check_before_buying", [])
    if what_to_check:
        with st.expander("✅ מה לבדוק לפני קנייה"):
            for item in what_to_check:
                st.write(f"• {item}")
    
    avoid_if = ownership.get("avoid_if", [])
    if avoid_if:
        st.warning("**להימנע אם**: " + ", ".join(avoid_if))
    
    good_if = ownership.get("good_candidate_if", [])
    if good_if:
        st.success("**מועמד טוב אם**: " + ", ".join(good_if))
    
    # Market supply
    market = car.get("market_supply", {})
    supply_level = market.get("level", "unknown")
    supply_basis = market.get("basis", "")
    st.markdown(f"**📊 היצע בשוק**: {supply_level}")
    if supply_basis:
        st.caption(f"בסיס: {supply_basis}")
    
    # Fit analysis
    fit_analysis = car.get("fit_analysis", {})
    why_fits = fit_analysis.get("why_it_fits", "")
    why_not = fit_analysis.get("why_it_may_not_fit", "")
    
    if why_fits or why_not:
        with st.expander("🎯 ניתוח התאמה"):
            if why_fits:
                st.success(f"**למה זה מתאים**: {why_fits}")
            if why_not:
                st.warning(f"**למה זה אולי לא מתאים**: {why_not}")
    
    best_for = fit_analysis.get("best_for", [])
    if best_for:
        st.info(f"**הכי טוב ל**: {', '.join(best_for)}")
    
    not_ideal = fit_analysis.get("not_ideal_for", [])
    if not_ideal:
        st.warning(f"**לא אידיאלי ל**: {', '.join(not_ideal)}")
    
    # Sources
    sources = car.get("sources", [])
    if sources:
        with st.expander("📚 מקורות"):
            for src in sources:
                if isinstance(src, dict):
                    title = src.get("title", "")
                    url = src.get("url", "")
                    supports = src.get("supports", "")
                    src_type = src.get("source_type", "")
                    
                    src_str = f"**{title}**" if title else "מקור"
                    if url:
                        src_str = f"[{src_str}]({url})"
                    if supports:
                        src_str += f" (תומך ב: {supports})"
                    if src_type:
                        src_str += f" [{src_type}]"
                    st.markdown(f"• {src_str}")
                else:
                    st.write(f"• {src}")
    else:
        st.warning("⚠️ לא נמצאו מקורות")
    
    st.markdown("---")
