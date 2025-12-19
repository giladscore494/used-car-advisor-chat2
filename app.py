# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה סופית (Modern Blue)
# שאלון 5 שלבים • הסברים בעברית • Gemini + Google Search (Grounding)
# + שינוי: חילוץ ואימות חיפוש/grounding מתוך ה-Response (לא רק בפרומפט)
# =========================================

import streamlit as st
import pandas as pd
import json, os, uuid
from datetime import datetime
import numpy as np

# --- Google GenAI (SDK החדש, עם Google Search) ---
from google import genai
from google.genai import types as genai_types


# --------------------------------------------------
# עיצוב כללי
# --------------------------------------------------
st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rubik:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root { --primary:#2259b4; --accent:#ff7a18; --ink:#0f172a; --sub:#64748b; }
html, body, [class*="css"] { font-family: 'Inter','Rubik',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial !important; }
h1,h2,h3 { color: var(--ink) }
.step { background:#fff; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,.06); padding:18px; margin-bottom:20px; }
.pill { display:inline-block; background:#eef2ff; color:#273c75; border-radius:9999px; padding:2px 10px; font-weight:600; margin-right:6px;}
.disclaimer { color:#a16207; background:#fffbeb; border:1px solid #fde68a; padding:8px 12px; border-radius:10px; }
.logo { height: 42px; margin-right:8px; vertical-align:middle; }
.topbar { display:flex; align-items:center; gap:10px; }
.small { color:#64748b; font-size:12px; }
.badge-ok { display:inline-block; padding:2px 10px; border-radius:9999px; background:#dcfce7; color:#166534; font-weight:700; }
.badge-warn { display:inline-block; padding:2px 10px; border-radius:9999px; background:#fef9c3; color:#854d0e; font-weight:700; }
.badge-bad { display:inline-block; padding:2px 10px; border-radius:9999px; background:#fee2e2; color:#991b1b; font-weight:700; }
.codebox { background:#0b1020; color:#e5e7eb; border-radius:12px; padding:12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:12px; overflow:auto; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Gemini config
# --------------------------------------------------
# שים לב: השם חייב להיות בדיוק כפי שה-API שלך מחזיר.
# אם לא בטוח השתמש באליאס שיש לך (gemini-flash-latest / gemini-pro-latest)
GEMINI_MODEL_ID = "gemini-3-flash-preview"


def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "⚠️ לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה."
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"שגיאה באתחול לקוח Gemini: {e}"


gemini_client, gemini_init_error = get_gemini_client()

# --------------------------------------------------
# State
# --------------------------------------------------
def init_state():
    for key in [
        "user_profile",
        "results_payload",
        "results_df",
        "methods_info",
        "search_info",
        "ui_step",
        "last_error",
        "fuel_price",
        "electricity_price",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.ui_step is None:
        st.session_state.ui_step = 0


# --------------------------------------------------
# פונקציות עזר
# --------------------------------------------------
def make_user_profile(
    budget_min,
    budget_max,
    years_range,
    fuels,
    gears,
    turbo_required,
    main_use,
    annual_km,
    driver_age,
    family_size,
    cargo_need,
    safety_required,
    trim_level,
    weights,
    body_style,
    driving_style,
    excluded_colors,
):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": (main_use or "").strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
        "family_size": family_size,
        "cargo_need": cargo_need,
        "safety_required": safety_required,
        "trim_level": trim_level,
        "weights": weights,
        "body_style": body_style,
        "driving_style": driving_style,
        "excluded_colors": excluded_colors,
    }


def clean_gemini_output(cars_raw):
    records, methods = [], []
    for car in cars_raw:
        if not isinstance(car, dict):
            continue
        record, method = {}, {}
        for k, v in car.items():
            if str(k).endswith("_method"):
                method[k] = v
            else:
                record[k] = v
        records.append(record)
        methods.append(method)
    return pd.DataFrame(records), methods


def normalize_car_values(df):
    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].replace(
            {
                "בנזין": "gasoline",
                "דיזל": "diesel",
                "היברידי": "hybrid",
                "דיזל היברידי": "hybrid-diesel",
                "חשמלי": "electric",
            }
        )
    if "gear" in df.columns:
        df["gear"] = df["gear"].replace(
            {
                "אוטומטי": "automatic",
                "אוטומטית": "automatic",
                "אוטומטי (DSG7)": "automatic",
                "אוטומטי (TCT)": "automatic",
                "אוטומטי (רובוטי)": "automatic",
                "ידני": "manual",
                "ידנית": "manual",
            }
        )
    if "turbo" in df.columns:
        df["turbo"] = df["turbo"].replace({"כן": True, "לא": False, True: True, False: False})
    return df


# --------------------------------------------------
# מיפויים
# --------------------------------------------------
fuel_map = {"בנזין": "gasoline", "היברידי": "hybrid", "דיזל היברידי": "hybrid-diesel", "דיזל": "diesel", "חשמלי": "electric"}
gear_map = {"אוטומטית": "automatic", "ידנית": "manual"}
turbo_map = {"לא משנה": "any", "כן": "yes", "לא": "no"}

fuel_map_he = {v: k for k, v in fuel_map.items()}
gear_map_he = {v: k for k, v in gear_map.items()}
turbo_map_he = {"yes": "כן", "no": "לא", "any": "לא משנה", True: "כן", False: "לא"}

column_map_he = {
    "brand": "מותג",
    "model": "דגם",
    "year": "שנה",
    "fuel": "דלק",
    "gear": "תיבה",
    "turbo": "טורבו",
    "engine_cc": 'נפח מנוע (סמ"ק)',
    "price_range_nis": "טווח מחיר (₪)",
    "avg_fuel_consumption": "צריכת דלק/אנרגיה",
    "annual_fee": "אגרה שנתית (₪)",
    "annual_energy_cost": "עלות דלק/חשמל שנתית (₪)",
    "total_annual_cost": "עלות כוללת שנתית (₪)",
    "reliability_score": "אמינות",
    "maintenance_cost": "עלות אחזקה (₪/שנה)",
    "safety_rating": "בטיחות",
    "insurance_cost": "עלות ביטוח (₪/שנה)",
    "resale_value": "שמירת ערך",
    "performance_score": "ביצועים",
    "comfort_features": "נוחות",
    "suitability": "התאמה",
    "market_supply": "היצע בשוק",
    "fit_score": "ציון התאמה (0–100)",
    "comparison_comment": "הערה השוואתית",
    "not_recommended_reason": "למה לא מומלץ",
}

method_map_he = {
    "fuel_method": "שיטת חישוב צריכת דלק/חשמל",
    "fee_method": "שיטת חישוב אגרה",
    "reliability_method": "שיטת חישוב אמינות",
    "maintenance_method": "שיטת חישוב עלות אחזקה",
    "safety_method": "שיטת חישוב בטיחות",
    "insurance_method": "שיטת חישוב ביטוח",
    "resale_method": "שיטת חישוב שמירת ערך",
    "performance_method": "שיטת חישוב ביצועים",
    "comfort_method": "שיטת חישוב נוחות",
    "suitability_method": "שיטת חישוב התאמה",
    "supply_method": "שיטת קביעת היצע",
}


# --------------------------------------------------
# Grounding extraction
# --------------------------------------------------
def _safe_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    for m in ("model_dump", "dict", "to_dict"):
        if hasattr(obj, m):
            try:
                return getattr(obj, m)()
            except Exception:
                pass
    try:
        return dict(obj.__dict__)
    except Exception:
        return None


def extract_grounding_info(resp) -> dict:
    info = {
        "has_grounding_metadata": False,
        "sources": [],
        "tool_signals": [],
        "raw_debug_available": False,
    }
    if resp is None:
        return info

    cand = None
    try:
        cands = getattr(resp, "candidates", None)
        if cands and len(cands) > 0:
            cand = cands[0]
    except Exception:
        cand = None

    gm = getattr(cand, "grounding_metadata", None) if cand is not None else getattr(resp, "grounding_metadata", None)
    if gm is not None:
        info["has_grounding_metadata"] = True
        gm_dict = _safe_to_dict(gm) or {}
        possible_chunks = gm_dict.get("grounding_chunks") or gm_dict.get("groundingChunks") or gm_dict.get("chunks") or []
        for ch in possible_chunks[:20]:
            chd = _safe_to_dict(ch) or {}
            web = chd.get("web") or chd.get("retrieved_context") or chd.get("retrievedContext") or {}
            webd = _safe_to_dict(web) or {}
            uri = webd.get("uri") or webd.get("url") or chd.get("uri") or chd.get("url")
            title = webd.get("title") or chd.get("title")
            if uri or title:
                info["sources"].append({"title": title or "", "uri": uri or ""})

        possible_supports = gm_dict.get("grounding_supports") or gm_dict.get("groundingSupports") or gm_dict.get("supports") or []
        if possible_supports:
            info["tool_signals"].append(f"grounding_supports={len(possible_supports)}")

    try:
        if cand is not None:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                info["raw_debug_available"] = True
                for p in parts:
                    pdict = _safe_to_dict(p) or {}
                    if "function_call" in pdict or "functionCall" in pdict:
                        info["tool_signals"].append("function_call_detected")
                    if "tool" in pdict or "tool_code" in pdict or "toolCode" in pdict:
                        info["tool_signals"].append("tool_part_detected")
    except Exception:
        pass

    info["tool_signals"] = sorted(list(set(info["tool_signals"])))
    return info


# --------------------------------------------------
# Gemini Call
# --------------------------------------------------
def call_gemini_with_search(profile: dict) -> dict:
    if gemini_client is None:
        return {"data": {"_error": gemini_init_error or "Gemini client unavailable."}, "grounding": {}, "raw_text": ""}

    prompt = f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

You are an independent automotive data analyst for the **Israeli used car market**.

🔴 CRITICAL INSTRUCTION: USE GOOGLE SEARCH TOOL
You MUST use the Google Search tool to verify:
- that the specific model and trim are actually sold in Israel
- realistic used prices in Israel (NIS)
- realistic fuel/energy consumption values
- common issues (DSG, reliability, recalls)

Hard constraints:
- Return only ONE top-level JSON object.
- JSON fields: "search_performed", "search_queries", "recommended_cars".
- search_performed: ALWAYS true (boolean).
- search_queries: array of the real Hebrew queries you would run in Google (max 6).
- All numeric fields must be pure numbers (no units, no text).

recommended_cars: array of 5–10 cars. EACH car MUST include:
  - brand
  - model
  - year
  - fuel
  - gear
  - turbo
  - engine_cc
  - price_range_nis
  - avg_fuel_consumption (+ fuel_method):
      * non-EV: km per liter (number only)
      * EV: kWh per 100 km (number only)
  - annual_fee (number only) + fee_method
  - reliability_score (1–10, number only) + reliability_method
  - maintenance_cost (number only) + maintenance_method
  - safety_rating (1–10, number only) + safety_method
  - insurance_cost (number only) + insurance_method
  - resale_value (1–10, number only) + resale_method
  - performance_score (1–10, number only) + performance_method
  - comfort_features (1–10, number only) + comfort_method
  - suitability (1–10, number only) + suitability_method
  - market_supply ("גבוה" / "בינוני" / "נמוך") + supply_method
  - fit_score (0–100, number only)
  - comparison_comment (Hebrew)
  - not_recommended_reason (Hebrew or null)

**All explanation fields (all *_method, comparison_comment, not_recommended_reason) MUST be in clean, easy Hebrew.**

IMPORTANT MARKET REALITY:
- לפני שאתה בוחר רכבים, תבדוק בזהירות בעזרת החיפוש שדגם כזה אכן נמכר בישראל, בתצורת מנוע וגיר שאתה מציג.
- אל תמציא דגמים או גרסאות שלא קיימים ביד 2 בישראל.
- מודלים שלא נמכרו כמעט / אין להם היצע – סמן "market_supply": "נמוך" והסבר בעברית.

Return ONLY raw JSON. Do not add any backticks or explanation text.
"""

    search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())

    config = genai_types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        tools=[search_tool],
        response_mime_type="application/json",
    )

    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=config,
        )

        raw_text = (getattr(resp, "text", "") or "").strip()
        grounding = extract_grounding_info(resp)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            data = {"_error": "JSON decode error from Gemini", "_raw": raw_text}

        return {"data": data, "grounding": grounding, "raw_text": raw_text}

    except Exception as e:
        return {"data": {"_error": f"Gemini call failed: {e}"}, "grounding": {}, "raw_text": ""}


# --------------------------------------------------
# Header
# --------------------------------------------------
def topbar():
    st.markdown(
        '<div class="topbar">'
        '<img src="https://em-content.zobj.net/source/microsoft-teams/363/automobile_1f697.png" class="logo"/>'
        '<div>'
        '<div style="font-weight:700;color:#0f172a;font-size:22px;">Car Advisor</div>'
        '<div style="color:#64748b;font-size:13px;">ייעוץ רכב • Smart Wizard</div>'
        '</div>'
        '<span class="pill">Modern</span><span class="pill">Fast</span><span class="pill">Grounded</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


init_state()
for _k in ["_step1", "_step2", "_step3", "_step4"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

topbar()

def nav_buttons(left_label="חזור", right_label="הבא", left_action=None, right_action=None, show_left=True, show_right=True):
    c1, c2 = st.columns([1, 1])
    with c1:
        if show_left:
            st.button(left_label, on_click=left_action, key=f"back_{st.session_state.ui_step}_{uuid.uuid4().hex}")
    with c2:
        if show_right:
            st.button(right_label, on_click=right_action, key=f"next_{st.session_state.ui_step}_{uuid.uuid4().hex}")


# --------------------------------------------------
# שלב 0
# --------------------------------------------------
if st.session_state.ui_step == 0:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.subheader("ברוך הבא ל-Car Advisor")
    st.write("מצא את הרכב המתאים לך בקלות. לחיצה על 'התחל' תוביל אותך לשאלון קצר.")
    if gemini_client is None:
        st.markdown(f'<div class="disclaimer">{gemini_init_error}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    def go_next():
        st.session_state.ui_step = 1

    nav_buttons(show_left=False, right_label="התחל", right_action=go_next)


# --------------------------------------------------
# שלב 1
# --------------------------------------------------
elif st.session_state.ui_step == 1:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 1: פרטים בסיסיים")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
    with c2:
        budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
    with c3:
        ymin, ymax = st.columns(2)
        with ymin:
            year_min = st.number_input("שנתון מינימום", 1990, datetime.now().year, 2015)
        with ymax:
            year_max = st.number_input("שנתון מקסימום", 1990, datetime.now().year, 2019)

    fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
    if "חשמלי" in fuels_he:
        st.info("נבחר דלק חשמלי — תיבת ההילוכים תוגדר כ'אוטומטית'.")
        gears_he = ["אוטומטית"]
    else:
        gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])

    turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)

    st.session_state._step1 = dict(
        budget_min=budget_min,
        budget_max=budget_max,
        year_min=year_min,
        year_max=year_max,
        fuels_he=fuels_he,
        gears_he=gears_he,
        turbo_choice_he=turbo_choice_he,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    def back():
        st.session_state.ui_step = 0

    def next():
        st.session_state.ui_step = 2

    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)


# --------------------------------------------------
# שלב 2  ✅ כאן התיקון
# --------------------------------------------------
elif st.session_state.ui_step == 2:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 2: שימוש וסגנון")

    c4, c5, c6 = st.columns([2, 1, 1])
    with c4:
        main_use = st.text_area("תיאור הרכב והשימוש בו", value="נסיעה יומיומית לעבודה וטיולים קצרים", height=100)
    with c5:
        # ✅ FIX: במקום st.n
        annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
    with c6:
        driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)

    c6a, c6b = st.columns(2)
    with c6a:
        license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
    with c6b:
        driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])

    cstyle1, cstyle2, cseats = st.columns([1, 1, 1])
    with cstyle1:
        body_style = st.selectbox("סגנון מרכב מועדף", ["כללי", "סדאן", "האצ'בק", "קרוסאובר/ג'יפון"])
    with cstyle2:
        driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח", "דינמי וספורטיבי"])
    with cseats:
        seats_choice = st.selectbox("מספר מקומות", ["4", "5", "5+"])

    excluded_colors = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="")
    excluded_colors = [c.strip() for c in excluded_colors.split(",") if c.strip()]

    st.session_state._step2 = dict(
        main_use=main_use,
        annual_km=annual_km,
        driver_age=driver_age,
        license_years=license_years,
        driver_gender=driver_gender,
        body_style=body_style,
        driving_style=driving_style,
        seats_choice=seats_choice,
        excluded_colors=excluded_colors,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    def back():
        st.session_state.ui_step = 1

    def next():
        st.session_state.ui_step = 3

    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)


# -------------------------------------------