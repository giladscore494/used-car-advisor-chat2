# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה ללא FitScore
# =========================================

import streamlit as st
import pandas as pd
import json, os
from datetime import datetime
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- Helpers --------
def init_state():
    for key in ["user_profile","validated_cars","methods_info"]:
        if key not in st.session_state:
            st.session_state[key] = None

# פרופיל משתמש בסיסי
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

# ניקוי פלט Gemini (השארת ציונים בלבד)
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

# -------- שלב 1: שאלון --------
init_state()
st.title("🚗 Car Advisor – ייעוץ רכב")

st.markdown("### שלב 1: שאלון")
col1, col2, col3 = st.columns([1,1,1])
with col1: budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
with col2: budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
with col3:
    ymin, ymax = st.columns(2)
    with ymin: year_min = st.number_input("שנתון מינימום", min_value=1990, max_value=datetime.now().year, value=2015)
    with ymax: year_max = st.number_input("שנתון מקסימום", min_value=1990, max_value=datetime.now().year, value=2019)

fuels = st.multiselect("סוגי דלק מועדפים", ["gasoline","hybrid","hybrid-diesel","diesel","electric"], default=["gasoline"])
gears = st.multiselect("תיבת הילוכים", ["automatic","manual"], default=["automatic"])
turbo_choice = st.selectbox("טורבו?", ["any","yes","no"], index=1)

c4, c5, c6 = st.columns([2,1,1])
with c4: main_use = st.text_input("שימוש עיקרי", value="נסיעה דינמית בסופשים חשוב רכב ספורטיבי ומאיץ טוב")
with c5: annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
with c6: driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)

# נתונים נוספים
family_size = st.selectbox("גודל משפחה", ["1-2","3-4","5+"])
cargo_need = st.selectbox("צורך בתא מטען", ["קטן","בינוני","גדול"])
safety_required = st.radio("חובה מערכות בטיחות אקטיביות?", ["כן","לא"])
trim_level = st.selectbox("רמת אבזור", ["בסיסי","סטנדרטי","עשיר"])

st.markdown("#### סדר עדיפויות (1-5)")
reliability_weight = st.slider("אמינות", 1, 5, 5)
resale_weight = st.slider("שמירת ערך", 1, 5, 3)
fuel_weight = st.slider("חיסכון בדלק", 1, 5, 4)
performance_weight = st.slider("ביצועים", 1, 5, 2)
comfort_weight = st.slider("נוחות", 1, 5, 3)

body_style = st.selectbox("סגנון מרכב מועדף", ["כללי","סדאן","האצ'בק","קרוסאובר/ג'יפון"])
driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח","דינמי וספורטיבי"])
excluded_colors = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="").split(",")

weights = {
    "reliability": reliability_weight,
    "resale": resale_weight,
    "fuel": fuel_weight,
    "performance": performance_weight,
    "comfort": comfort_weight,
}

profile = make_user_profile(
    budget_min, budget_max, [year_min, year_max],
    fuels, gears, turbo_choice, main_use, annual_km, driver_age,
    family_size, cargo_need, safety_required, trim_level,
    weights, body_style, driving_style, excluded_colors
)

st.session_state.user_profile = profile

# -------- שלב 2: Gemini --------
st.markdown("### שלב 2: Gemini – המלצות ראשוניות")
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
else:
    genai.configure(api_key=api_key)
    model_name = "models/gemini-2.5-pro"
    model = genai.GenerativeModel(model_name)

    if st.button("🚀 בקש המלצות מגימניי"):
        prompt = f"""
        אני צריך המלצות לרכבים ללקוח ישראלי. זה הפרופיל:
        {json.dumps(profile, ensure_ascii=False, indent=2)}

        דרישות לפלט:
        1. החזר JSON יחיד עם שלושה שדות: "search_performed", "search_queries", "recommended_cars".
        2. search_performed: True אם בוצע חיפוש אינטרנטי, אחרת False.
        3. search_queries: מערך עם מחרוזות החיפוש שבוצעו בפועל.
        4. recommended_cars: מערך של 5–10 רכבים. כל רכב חייב לכלול:
           - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
           - reliability_score (מספר שלם 1–10 בלבד) + reliability_method
           - maintenance_cost (₪ לשנה, מספר בלבד) + maintenance_method
           - safety_rating (מספר שלם 1–10 בלבד) + safety_method
           - insurance_cost (₪ לשנה, מספר בלבד) + insurance_method
           - resale_value (מספר שלם 1–10 בלבד) + resale_method
           - performance_score (מספר שלם 1–10 בלבד) + performance_method
           - comfort_features (מספר שלם 1–10 בלבד) + comfort_method
           - suitability (מספר שלם 1–10 בלבד) + suitability_method
        5. חובה להחזיר **אך ורק מספרים** עבור כל הציונים (בלי טקסטים כמו "בינוני" או "גבוה").
        6. חובה להחזיר רכבים שנמכרים בפועל בישראל בלבד.
        """

        with st.spinner("פונה לגימניי..."):
            try:
                resp = model.generate_content(prompt)
                text = resp.candidates[0].content.parts[0].text.strip()

                if text.startswith("```"):
                    text = text.strip("`")
                    text = text.replace("json\n", "").replace("json", "").strip()

                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    st.error("⚠️ גימניי לא החזיר JSON תקין. להלן הפלט:")
                    st.code(text)
                    parsed = {}

            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
                parsed = {}

        if parsed and "recommended_cars" in parsed:
            search_performed = parsed.get("search_performed", False)
            search_queries = parsed.get("search_queries", [])
            if search_performed and search_queries:
                st.info("✅ בוצע חיפוש אינטרנטי לנתוני שוק עדכניים.")
            else:
                st.warning("⚠️ לא ברור אם בוצע חיפוש חי. ייתכן שהנתונים חלקיים.")

            cars_to_process = parsed["recommended_cars"]
            results_df, methods_info = clean_gemini_output(cars_to_process)

            if not results_df.empty:
                st.session_state.validated_cars = results_df
                st.session_state.methods_info = methods_info

                st.success(f"✅ התקבלו {len(results_df)} רכבים מגימניי.")
                st.dataframe(results_df.reset_index(drop=True))

                st.markdown("### 📖 הסברים לכל פרמטר")
                for i, method in enumerate(methods_info, 1):
                    with st.expander(f"🔎 רכב {i} – הסברים"):
                        for k, v in method.items():
                            st.write(f"- **{k}:** {v}")
            else:
                st.error("⚠️ לא נמצאו רכבים בפלט.")
        else:
            st.error("⚠️ הפלט מגימניי לא כלל שדה 'recommended_cars'.")
