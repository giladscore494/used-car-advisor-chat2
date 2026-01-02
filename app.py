
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה סופית (Modern Blue)
# שאלון 5 שלבים • הסברים בעברית • Gemini 3 + Google Search
# =========================================

import streamlit as st
import pandas as pd
import json, os, uuid, re, logging
from datetime import datetime
import numpy as np

# --- Google GenAI (SDK החדש, עם Google Search) ---
from google import genai
from google.genai import types as genai_types

# --- Israeli Car Market Data ---
try:
    from car_models_dict import israeli_car_market_full_compilation
    logging.info(f"[CAR_DICT] ✅ Loaded {len(israeli_car_market_full_compilation)} manufacturers")
except Exception as e:
    logging.warning(f"[CAR_DICT] ⚠️ Failed to load car_models_dict: {e}")
    israeli_car_market_full_compilation = {}

# --------------------------------------------------
# עיצוב כללי
# --------------------------------------------------
st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Gemini 3 Pro config (SDK החדש)
# --------------------------------------------------
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
# Validation Constants
# --------------------------------------------------
MIN_MODEL_NAME_LENGTH = 2  # Minimum characters for model name validation
SAMPLE_BRANDS_COUNT = 10   # Number of brands to show in AI prompt sample

# --------------------------------------------------
# פונקציות עזר (הלוגיקה המקורית נשמרה)
# --------------------------------------------------
def init_state():
    for key in ["user_profile","validated_cars","methods_info","fuel_price","electricity_price","ui_step","results_df","search_info"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.ui_step is None:
        st.session_state.ui_step = 0

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age,
                      family_size, cargo_need, safety_required,
                      trim_level, weights, body_style, driving_style, excluded_colors):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
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
            if k.endswith("_method"):
                method[k] = v
            else:
                record[k] = v
        records.append(record)
        methods.append(method)
    return pd.DataFrame(records), methods

def normalize_car_values(df):
    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].replace({
            "בנזין": "gasoline",
            "דיזל": "diesel",
            "היברידי": "hybrid",
            "דיזל היברידי": "hybrid-diesel",
            "חשמלי": "electric"
        })
    if "gear" in df.columns:
        df["gear"] = df["gear"].replace({
            "אוטומטי": "automatic",
            "אוטומטית": "automatic",
            "אוטומטי (DSG7)": "automatic",
            "אוטומטי (TCT)": "automatic",
            "אוטומטי (רובוטי)": "automatic",
            "ידני": "manual",
            "ידנית": "manual"
        })
    if "turbo" in df.columns:
        df["turbo"] = df["turbo"].replace({"כן": True, "לא": False, True: True, False: False})
    return df

# Create brand lookup cache for performance
_brand_lookup_cache = None

def _get_brand_lookup_cache():
    """
    Get or create cached brand lookup dictionary.
    
    Note: This cache is per-process (shared across all Streamlit sessions/users).
    In Streamlit's execution model, each session runs in a single thread but
    shares the Python process, so this cache benefits all users simultaneously.
    For multi-threaded applications outside Streamlit, use threading.Lock.
    """
    global _brand_lookup_cache
    if _brand_lookup_cache is None and israeli_car_market_full_compilation:
        _brand_lookup_cache = {
            brand.lower(): brand 
            for brand in israeli_car_market_full_compilation.keys()
        }
    return _brand_lookup_cache or {}

def validate_car_in_israeli_market(brand: str, model: str) -> tuple[bool, str]:
    """
    Validate if a car model exists in the Israeli market database.
    Returns: (is_valid, message)
    """
    if not israeli_car_market_full_compilation:
        # If dictionary not loaded, return neutral status (not validated, not failed)
        # This is logged at startup, so we don't repeat the warning
        return True, "⚠️ לא ניתן לאמת - מאגר לא נטען"
    
    # Normalize inputs
    brand_normalized = brand.strip().lower()
    model_normalized = model.strip().lower()
    
    # Minimum length check to avoid false positives
    if len(model_normalized) < MIN_MODEL_NAME_LENGTH:
        return False, f"⚠️ שם דגם קצר מדי: '{model}'"
    
    # Use cached brand lookup for performance
    brand_cache = _get_brand_lookup_cache()
    matching_brand = brand_cache.get(brand_normalized)
    
    if not matching_brand:
        return False, f"⚠️ מותג '{brand}' לא נמצא במאגר הישראלי"
    
    # Check if model exists with improved matching
    models_list = israeli_car_market_full_compilation[matching_brand]
    
    for db_model in models_list:
        # Extract model name without year range
        # Assume year range is always at the end in format (YYYY-YYYY) or (YYYY)
        # This is safer than just splitting on first '(' which could be in model name
        match = re.match(r'^(.+?)\s*\((\d{4}(?:-\d{4})?)\)$', db_model.strip())
        if match:
            db_model_name = match.group(1).strip().lower()
        else:
            # Fallback to old method if format doesn't match
            db_model_name = db_model.split('(')[0].strip().lower()
        
        # Improved matching logic:
        # 1. Exact match
        if model_normalized == db_model_name:
            return True, f"✓ נמצא במאגר: {db_model}"
        
        # 2. Model name appears at the start of db_model_name
        if db_model_name.startswith(model_normalized):
            return True, f"✓ נמצא במאגר: {db_model}"
        
        # 3. db_model_name appears at the start of model name (for variations like "i30 N")
        if model_normalized.startswith(db_model_name):
            return True, f"✓ נמצא במאגר: {db_model}"
        
        # 4. Handle common variations (e.g., "rav4" vs "rav-4")
        model_no_spaces = model_normalized.replace(' ', '').replace('-', '')
        db_model_no_spaces = db_model_name.replace(' ', '').replace('-', '')
        if model_no_spaces == db_model_no_spaces:
            return True, f"✓ נמצא במאגר: {db_model}"
    
    return False, f"⚠️ דגם '{model}' של {brand} לא נמצא במאגר הישראלי"

def add_market_validation_to_results(df):
    """Add validation column to results dataframe"""
    if df.empty or not israeli_car_market_full_compilation:
        return df
    
    validations = []
    for _, row in df.iterrows():
        brand = row.get('brand', '')
        model = row.get('model', '')
        is_valid, msg = validate_car_in_israeli_market(brand, model)
        validations.append({'is_valid': is_valid, 'message': msg})
    
    df['market_validation'] = [v['message'] for v in validations]
    return df

# --------------------------------------------------
# מיפויים
# --------------------------------------------------
fuel_map = {"בנזין":"gasoline","היברידי":"hybrid","דיזל היברידי":"hybrid-diesel","דיזל":"diesel","חשמלי":"electric"}
gear_map = {"אוטומטית":"automatic","ידנית":"manual"}
turbo_map = {"לא משנה":"any","כן":"yes","לא":"no"}

fuel_map_he = {v:k for k,v in fuel_map.items()}
gear_map_he = {v:k for k,v in gear_map.items()}
turbo_map_he = {"yes":"כן","no":"לא","any":"לא משנה",True:"כן",False:"לא"}

column_map_he = {
    "brand":"מותג","model":"דגם","year":"שנה","fuel":"דלק","gear":"תיבה","turbo":"טורבו","engine_cc":"נפח מנוע (סמ\"ק)",
    "price_range_nis":"טווח מחיר (₪)","avg_fuel_consumption":"צריכת דלק ממוצעת (ק\"מ/ל')","annual_fee":"אגרה שנתית (₪)",
    "annual_energy_cost":"עלות דלק שנתית (₪)","total_annual_cost":"עלות כוללת שנתית (₪)",
    "reliability_score":"אמינות","maintenance_cost":"עלות אחזקה (₪/שנה)","safety_rating":"בטיחות",
    "insurance_cost":"עלות ביטוח (₪/שנה)","resale_value":"שמירת ערך","performance_score":"ביצועים",
    "comfort_features":"נוחות","suitability":"התאמה","market_supply":"היצע בשוק","market_validation":"אימות שוק ישראלי"
}

method_map_he = {
    "fuel_method":"שיטת חישוב צריכת דלק/חשמל","fee_method":"שיטת חישוב אגרה","reliability_method":"שיטת חישוב אמינות",
    "maintenance_method":"שיטת חישוב עלות אחזקה","safety_method":"שיטת חישוב בטיחות","insurance_method":"שיטת חישוב ביטוח",
    "resale_method":"שיטת חישוב שמירת ערך","performance_method":"שיטת חישוב ביצועים","comfort_method":"שיטת חישוב נוחות",
    "suitability_method":"שיטת חישוב התאמה","supply_method":"שיטת קביעת היצע"
}

# --------------------------------------------------
# Gemini Call – כמו בבנץ' (חיפוש אמיתי)
# --------------------------------------------------
def call_gemini_with_search(profile: dict) -> dict:
    """
    קריאה ל-Gemini 3 Pro עם Google Search מופעל ו-output כ-JSON בלבד.
    """
    if gemini_client is None:
        return {"_error": gemini_init_error or "Gemini client unavailable."}

    # Prepare Israeli market models reference
    market_models_sample = ""
    if israeli_car_market_full_compilation:
        # Show top manufacturers with their model count as reference
        top_brands = list(israeli_car_market_full_compilation.keys())[:SAMPLE_BRANDS_COUNT]
        market_models_sample = "\n".join([
            f"  - {brand}: {len(israeli_car_market_full_compilation[brand])} models"
            for brand in top_brands
        ])
        remaining_count = len(israeli_car_market_full_compilation) - SAMPLE_BRANDS_COUNT
        market_models_sample = f"\n\nISRAELI MARKET MODELS REFERENCE (sample):\n{market_models_sample}\n... and {remaining_count} more manufacturers"

    prompt = f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

You are an independent automotive data analyst for the **Israeli used car market**.
{market_models_sample}

🔴 CRITICAL INSTRUCTION: USE GOOGLE SEARCH TOOL
You MUST use the Google Search tool to verify:
- that the specific model and trim are actually sold in Israel
- realistic used prices in Israel (NIS)
- realistic fuel/energy consumption values
- common issues (DSG, reliability, recalls)

🔴 CRITICAL: ONLY recommend models that are actually sold in the Israeli market.
Reference the Israeli market models database above. Do NOT invent models or trims that don't exist.

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
  - annual_fee (₪/year, number only) + fee_method
  - reliability_score (1–10, number only) + reliability_method
  - maintenance_cost (₪/year, number only) + maintenance_method
  - safety_rating (1–10, number only) + safety_method
  - insurance_cost (₪/year, number only) + insurance_method
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
- השתמש במידע מהמאגר הישראלי לעיל כנקודת ייחוס ראשונית.
- מודלים שלא נמכרו כמעט / אין להם היצע – סמן "market_supply": "נמוך" והסבר בעברית.

Return ONLY raw JSON. Do not add any backticks or explanation text.
"""

    # Search tool
    search_tool = genai_types.Tool(
        google_search=genai_types.GoogleSearch()
    )

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
        text = getattr(resp, "text", "") or ""
        text = text.strip()
        # רק JSON נקי – אין טיפול ב-``` כי הגדרנו response_mime_type
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_error": "JSON decode error from Gemini", "_raw": text}
    except Exception as e:
        return {"_error": f"Gemini call failed: {e}"}

# --------------------------------------------------
# Init + Header
# --------------------------------------------------
def topbar():
    st.markdown(
        '<div class="topbar"><img src="https://em-content.zobj.net/source/microsoft-teams/363/automobile_1f697.png" class="logo"/>'
        '<div><div style="font-weight:700;color:#0f172a;font-size:22px;">Car Advisor</div>'
        '<div style="color:#64748b;font-size:13px;">ייעוץ רכב • Smart Wizard</div></div>'
        '<span class="pill">Modern</span><span class="pill">Fast</span><span class="pill">Clean</span></div>',
        unsafe_allow_html=True)
    st.markdown("---")

init_state()
for _k in ["_step1","_step2","_step3","_step4"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
topbar()

# --------------------------------------------------
# פונקציית ניווט (מזהים ייחודיים כדי למנוע DuplicateKey)
# --------------------------------------------------
def nav_buttons(left_label="חזור", right_label="הבא",
                left_action=None, right_action=None,
                show_left=True, show_right=True):
    c1,c2 = st.columns([1,1])
    with c1:
        if show_left:
            st.button(left_label, on_click=left_action, key=f"back_{st.session_state.ui_step}_{uuid.uuid4().hex}")
    with c2:
        if show_right:
            st.button(right_label, on_click=right_action, key=f"next_{st.session_state.ui_step}_{uuid.uuid4().hex}")

# --------------------------------------------------
# שלב 0 – פתיחה
# --------------------------------------------------
if st.session_state.ui_step == 0:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.subheader("ברוך הבא ל-Car Advisor")
    st.write("מצא את הרכב המתאים לך בקלות. לחיצה על 'התחל' תוביל אותך לשאלון קצר.")
    st.markdown('</div>', unsafe_allow_html=True)
    def go_next(): st.session_state.ui_step = 1
    nav_buttons(show_left=False, right_label="התחל", right_action=go_next)

# --------------------------------------------------
# שלב 1 – פרטים בסיסיים
# --------------------------------------------------
elif st.session_state.ui_step == 1:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 1: פרטים בסיסיים")
    c1, c2, c3 = st.columns([1,1,1])
    with c1: budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
    with c2: budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
    with c3:
        ymin, ymax = st.columns(2)
        with ymin: year_min = st.number_input("שנתון מינימום", 1990, datetime.now().year, 2015)
        with ymax: year_max = st.number_input("שנתון מקסימום", 1990, datetime.now().year, 2019)
    fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
    if "חשמלי" in fuels_he:
        st.info("נבחר דלק חשמלי — תיבת ההילוכים תוגדר כ'אוטומטית'.")
        gears_he = ["אוטומטית"]
    else:
        gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])
    turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)
    st.session_state._step1 = dict(
        budget_min=budget_min, budget_max=budget_max,
        year_min=year_min, year_max=year_max,
        fuels_he=fuels_he, gears_he=gears_he, turbo_choice_he=turbo_choice_he
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 0
    def next(): st.session_state.ui_step = 2
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 2 – שימוש וסגנון
# --------------------------------------------------
elif st.session_state.ui_step == 2:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 2: שימוש וסגנון")
    c4, c5, c6 = st.columns([2,1,1])
    with c4:
        main_use = st.text_area("תיאור הרכב והשימוש בו", value="נסיעה יומיומית לעבודה וטיולים קצרים", height=100)
    with c5:
        annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
    with c6:
        driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)
    c6a, c6b = st.columns(2)
    with c6a: license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
    with c6b: driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])
    cstyle1, cstyle2, cseats = st.columns([1,1,1])
    with cstyle1: body_style = st.selectbox("סגנון מרכב מועדף", ["כללי","סדאן","האצ'בק","קרוסאובר/ג'יפון"])
    with cstyle2: driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח","דינמי וספורטיבי"])
    with cseats: seats_choice = st.selectbox("מספר מקומות", ["4","5","5+"] )
    excluded_colors = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="").split(",")
    st.session_state._step2 = dict(
        main_use=main_use, annual_km=annual_km, driver_age=driver_age,
        license_years=license_years, driver_gender=driver_gender,
        body_style=body_style, driving_style=driving_style,
        seats_choice=seats_choice, excluded_colors=excluded_colors
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 1
    def next(): st.session_state.ui_step = 3
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 3 – סדר עדיפויות
# --------------------------------------------------
elif st.session_state.ui_step == 3:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 3: סדר עדיפויות")
    st.markdown("#### בחר דירוג לכל קטגוריה (1–5)")
    reliability_weight = st.slider("אמינות", 1, 5, 5)
    resale_weight = st.slider("שמירת ערך", 1, 5, 3)
    fuel_weight = st.slider("חיסכון בדלק", 1, 5, 4)
    performance_weight = st.slider("ביצועים", 1, 5, 2)
    comfort_weight = st.slider("נוחות", 1, 5, 3)
    weights = {
        "reliability": reliability_weight,
        "resale": resale_weight,
        "fuel": fuel_weight,
        "performance": performance_weight,
        "comfort": comfort_weight,
    }
    st.session_state._step3 = dict(weights=weights)
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 2
    def next(): st.session_state.ui_step = 4
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 4 – פרטים נוספים
# --------------------------------------------------
elif st.session_state.ui_step == 4:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 4: פרטים נוספים")
    insurance_history = st.text_input("עבר ביטוחי", value="שנתיים ללא תביעות")
    violations = st.selectbox("דוחות/שלילות", ["אין", "שלילה בעבר", "נקודות פעילות"])
    cfam, ccargo, csafety, ctrim = st.columns([1,1,1,1])
    with cfam: family_size = st.selectbox("גודל משפחה", ["1-2","3-4","5+"])
    with ccargo: cargo_need = st.selectbox("צורך בתא מטען", ["קטן","בינוני","גדול"])
    with csafety: safety_required = st.radio("חובה מערכות בטיחות אקטיביות?", ["כן","לא"])
    with ctrim: trim_level = st.selectbox("רמת אבזור", ["בסיסי","סטנדרטי","עשיר"])
    consider_supply = st.radio("האם להתחשב בהיצע בשוק?", ["כן","לא"], index=0)
    cfp, cep = st.columns([1,1])
    with cfp: fuel_price = st.number_input("מחיר ליטר דלק (₪)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    with cep: electricity_price = st.number_input("מחיר חשמל לקוט״ש (₪)", min_value=0.1, max_value=5.0, value=0.65, step=0.01)
    st.session_state._step4 = dict(
        insurance_history=insurance_history, violations=violations,
        family_size=family_size, cargo_need=cargo_need, safety_required=safety_required,
        trim_level=trim_level, consider_supply=consider_supply,
        fuel_price=fuel_price, electricity_price=electricity_price
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 3
    def next(): st.session_state.ui_step = 5
    nav_buttons(left_label="חזור", right_label="המשך לייעוץ", left_action=back, right_action=next)

# --------------------------------------------------
# שלב 5 – ייעוץ ותוצאות (עם Gemini 3 + Google Search)
# --------------------------------------------------
elif st.session_state.ui_step == 5:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 5: קבלת ייעוץ ותוצאות")

    s1 = st.session_state._step1
    s2 = st.session_state._step2
    s3 = st.session_state._step3
    s4 = st.session_state._step4
    if not all([s1, s2, s3, s4]):
        st.error("חסרים נתונים בשלבים קודמים. חזור אחורה והשלם.")
    else:
        # בניית פרופיל (לוגיקה מקורית 1:1)
        fuels = [fuel_map[f] for f in (s1["fuels_he"] or [])]
        gears = [gear_map[g] for g in (s1["gears_he"] or [])]
        turbo_choice = turbo_map[s1["turbo_choice_he"]]
        weights = s3["weights"]

        profile = make_user_profile(
            s1["budget_min"], s1["budget_max"], [s1["year_min"], s1["year_max"]],
            fuels, gears, turbo_choice, s2["main_use"], s2["annual_km"], s2["driver_age"],
            s4["family_size"], s4["cargo_need"], s4["safety_required"], s4["trim_level"],
            weights, s2["body_style"], s2["driving_style"], s2["excluded_colors"]
        )
        profile["license_years"] = s2["license_years"]
        profile["driver_gender"] = s2["driver_gender"]
        profile["insurance_history"] = s4["insurance_history"]
        profile["violations"] = s4["violations"]
        profile["consider_market_supply"] = (s4["consider_supply"] == "כן")
        profile["fuel_price_nis_per_liter"] = s4["fuel_price"]
        profile["electricity_price_nis_per_kwh"] = s4["electricity_price"]
        profile["seats"] = s2["seats_choice"]  # מידע בלבד

        st.session_state.user_profile = profile
        st.session_state.fuel_price = s4["fuel_price"]
        st.session_state.electricity_price = s4["electricity_price"]

        if gemini_client is None:
            st.warning(gemini_init_error or "⚠️ לא ניתן להתחבר ל-Gemini.")
        else:
            if st.button("🚀 בקש המלצות מג׳מיני (Gemini 3 + חיפוש רשת)"):
                with st.spinner("פונה לג׳מיני 3 פרו עם חיפוש חי..."):
                    parsed = call_gemini_with_search(profile)

                if parsed.get("_error"):
                    st.error(f"⚠️ שגיאה מג׳מיני: {parsed.get('_error')}")
                    raw = parsed.get("_raw")
                    if raw:
                        st.code(raw, language="json")
                elif parsed and "recommended_cars" in parsed:
                    search_performed = parsed.get("search_performed", False)
                    search_queries = parsed.get("search_queries", [])
                    st.session_state.search_info = {"search_performed": search_performed, "search_queries": search_queries}
                    if search_performed and search_queries:
                        st.info(f"✅ בוצע חיפוש אינטרנטי. שאילתות (סה\"כ {len(search_queries)}):")
                        st.code("\n".join(search_queries))
                    else:
                        st.warning("⚠️ לא ברור אם בוצע חיפוש חי. ייתכן שהנתונים חלקיים.")

                    results_df, methods_info = clean_gemini_output(parsed["recommended_cars"])
                    if not results_df.empty:
                        # Normalize
                        results_df = normalize_car_values(results_df)
                        
                        # Add market validation
                        results_df = add_market_validation_to_results(results_df)

                        if "avg_fuel_consumption" not in results_df.columns:
                            st.error("חסר שדה avg_fuel_consumption בפלט.")
                            st.stop()

                        # חישובי אנרגיה
                        is_ev = results_df["fuel"].str.lower().eq("electric")
                        km_per_liter = results_df["avg_fuel_consumption"].where(~is_ev, np.nan).replace(0, np.nan)
                        kwh_per_100km = results_df["avg_fuel_consumption"].where(is_ev, np.nan)
                        annual_km = profile["annual_km"]
                        fuel_price = st.session_state.fuel_price or 7.0
                        elec_price = st.session_state.electricity_price or 0.65

                        fuel_cost = (annual_km / km_per_liter) * fuel_price
                        elec_cost = (annual_km / 100.0) * kwh_per_100km * elec_price
                        results_df["annual_energy_cost"] = np.where(is_ev, elec_cost, fuel_cost)
                        # תאימות לעבר – לא מציגים את annual_fuel_cost באנגלית
                        results_df["annual_fuel_cost"] = results_df["annual_energy_cost"]

                        for col in ["maintenance_cost", "insurance_cost", "annual_fee"]:
                            if col not in results_df.columns:
                                results_df[col] = 0.0

                        results_df["total_annual_cost"] = (
                            results_df["annual_energy_cost"].fillna(0) +
                            results_df["maintenance_cost"].fillna(0) +
                            results_df["insurance_cost"].fillna(0) +
                            results_df["annual_fee"].fillna(0)
                        )

                        # כותרות צריכה בעברית לפי EV/דלק
                        if results_df["fuel"].str.lower().eq("electric").any():
                            column_map_he["avg_fuel_consumption"] = "צריכת חשמל (קוט\"ש/100 ק\"מ)"
                            column_map_he["annual_energy_cost"] = "עלות חשמל שנתית (₪)"
                        else:
                            column_map_he["avg_fuel_consumption"] = "צריכת דלק ממוצעת (ק\"מ/ל')"
                            column_map_he["annual_energy_cost"] = "עלות דלק שנתית (₪)"

                        # טבלת תצוגה בעברית – בלי העמודה האנגלית
                        results_df_display = results_df.copy()
                        if "annual_fuel_cost" in results_df_display.columns:
                            results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])
                        results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
                        results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
                        results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
                        results_df_display = results_df_display.rename(columns=column_map_he)

                        st.session_state.results_df = results_df
                        st.session_state.methods_info = methods_info
                        st.success(f"✅ התקבלו {len(results_df)} רכבים מג׳מיני 3.")
                    else:
                        st.error("⚠️ לא נמצאו רכבים בפלט.")
                else:
                    st.error("⚠️ לא התקבל מפתח 'recommended_cars' בפלט מג׳מיני.")

        # הצגת תוצאות אם קיימות
        results_df = st.session_state.results_df
        methods_info = st.session_state.methods_info or []
        if results_df is not None and not results_df.empty:
            results_df_display = results_df.copy()
            # להבטיח שלא מציגים את העמודה האנגלית
            if "annual_fuel_cost" in results_df_display.columns:
                results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])
            results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
            results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
            results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
            results_df_display = results_df_display.rename(columns=column_map_he)

            st.markdown('<div class="disclaimer">⚠️ הנתונים הם הערכה גסה של AI; יש לאמת לפני קנייה.</div>', unsafe_allow_html=True)
            st.dataframe(results_df_display.reset_index(drop=True), use_container_width=True)

            # גרף עלות כוללת שנתית
            st.markdown("### 📊 השוואת עלות כוללת שנתית")
            chart_df = results_df_display[["מותג","דגם","שנה","עלות כוללת שנתית (₪)"]].copy()
            chart_df["רכב"] = chart_df["מותג"] + " " + chart_df["דגם"] + " " + chart_df["שנה"].astype(str)
            chart_df = chart_df.set_index("רכב")
            st.bar_chart(chart_df["עלות כוללת שנתית (₪)"])

            # הסברים מפורטים בעברית לכל רכב
            st.markdown("### 📝 הסברים מפורטים לכל רכב")
            for i, row in results_df_display.iterrows():
                car_name = f"{row['מותג']} {row['דגם']} {row['שנה']}"
                with st.expander(f"📝 הסבר מפורט על {car_name}"):
                    st.caption(f"דלק: {row['דלק']} | תיבה: {row['תיבה']} | טורבו: {row['טורבו']}")
                    if "טווח מחיר (₪)" in row:
                        st.write(f"**טווח מחיר:** {row['טווח מחיר (₪)']}")
                    if "עלות כוללת שנתית (₪)" in row:
                        try:
                            st.write(f"**עלות שנתית:** {float(row['עלות כוללת שנתית (₪)']):,.0f} ₪")
                        except Exception:
                            st.write(f"**עלות שנתית:** {row['עלות כוללת שנתית (₪)']} ₪")
                    method = methods_info[i] if i < len(methods_info) else {}
                    if method:
                        for k, v in method.items():
                            name = method_map_he.get(k, k)
                            st.write(f"- **{name}:** {v}")
                    else:
                        st.write("אין הסברים מפורטים זמינים לפריט זה.")

    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 4
    nav_buttons(left_label="חזור", right_label="הבא", left_action=back, right_action=None, show_right=False)
