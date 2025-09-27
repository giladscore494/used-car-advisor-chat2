# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה מלאה עם צריכת דלק (ק"מ/ל'), אגרה שנתית, ועלות דלק שנתית
# =========================================

import streamlit as st
import pandas as pd
import json, os
from datetime import datetime
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- קבועים --------
FUEL_PRICE_NIS = 7.0  # מחיר ליטר דלק ממוצע (₪)

# -------- Helpers --------
def init_state():
    for key in ["user_profile","validated_cars","methods_info"]:
        if key not in st.session_state:
            st.session_state[key] = None

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

# -------- מיפויים --------
fuel_map = {
    "בנזין": "gasoline",
    "היברידי": "hybrid",
    "דיזל היברידי": "hybrid-diesel",
    "דיזל": "diesel",
    "חשמלי": "electric"
}
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
    "engine_cc": "נפח מנוע (סמ\"ק)",
    "price_range_nis": "טווח מחיר (₪)",
    "avg_fuel_consumption": "צריכת דלק ממוצעת (ק\"מ/ל')",
    "annual_fee": "אגרה שנתית (₪)",
    "annual_fuel_cost": "עלות דלק שנתית (₪)",
    "reliability_score": "אמינות",
    "maintenance_cost": "עלות אחזקה (₪/שנה)",
    "safety_rating": "בטיחות",
    "insurance_cost": "עלות ביטוח (₪/שנה)",
    "resale_value": "שמירת ערך",
    "performance_score": "ביצועים",
    "comfort_features": "נוחות",
    "suitability": "התאמה"
}

method_map_he = {
    "fuel_method": "שיטת חישוב צריכת דלק",
    "fee_method": "שיטת חישוב אגרה",
    "reliability_method": "שיטת חישוב אמינות",
    "maintenance_method": "שיטת חישוב עלות אחזקה",
    "safety_method": "שיטת חישוב בטיחות",
    "insurance_method": "שיטת חישוב ביטוח",
    "resale_method": "שיטת חישוב שמירת ערך",
    "performance_method": "שיטת חישוב ביצועים",
    "comfort_method": "שיטת חישוב נוחות",
    "suitability_method": "שיטת חישוב התאמה"
}

# -------- שלב 1 --------
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

fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])
turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)

fuels = [fuel_map[f] for f in fuels_he]
gears = [gear_map[g] for g in gears_he]
turbo_choice = turbo_map[turbo_choice_he]

c4, c5, c6 = st.columns([2,1,1])
with c4: main_use = st.text_input("שימוש עיקרי", value="נסיעה יומיומית")
with c5: annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
with c6: driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)

c6a, c6b = st.columns(2)
with c6a: license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
with c6b: driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])

insurance_history = st.text_input("עבר ביטוחי", value="שנתיים ללא תביעות")
violations = st.selectbox("דוחות/שלילות", ["אין", "שלילה בעבר", "נקודות פעילות"])

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
profile["license_years"] = license_years
profile["driver_gender"] = driver_gender
profile["insurance_history"] = insurance_history
profile["violations"] = violations

st.session_state.user_profile = profile

# -------- שלב 2 --------
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
        - החזר JSON עם שלושה שדות: "search_performed", "search_queries", "recommended_cars".
        - כל רכב חייב לכלול:
          brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
          avg_fuel_consumption (ק\"מ/ל', מספר בלבד) + fuel_method
          annual_fee (₪ לשנה, מספר בלבד) + fee_method
          reliability_score (1–10) + reliability_method
          maintenance_cost (₪/שנה) + maintenance_method
          safety_rating (1–10) + safety_method
          insurance_cost (₪/שנה) + insurance_method
          resale_value (1–10) + resale_method
          performance_score (1–10) + performance_method
          comfort_features (1–10) + comfort_method
          suitability (1–10) + suitability_method
        """

        with st.spinner("פונה לגימניי..."):
            try:
                resp = model.generate_content(prompt)
                text = resp.candidates[0].content.parts[0].text.strip()
                if text.startswith("```"):
                    text = text.strip("`").replace("json\n", "").replace("json", "").strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    st.error("⚠️ גימניי לא החזיר JSON תקין.")
                    st.code(text)
                    parsed = {}
            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
                parsed = {}

        if parsed and "recommended_cars" in parsed:
            cars_to_process = parsed["recommended_cars"]
            results_df, methods_info = clean_gemini_output(cars_to_process)

            if not results_df.empty:
                # --- חישוב עלות דלק שנתית ---
                results_df["annual_fuel_cost"] = (
                    profile["annual_km"] / results_df["avg_fuel_consumption"].replace(0, 1)
                ) * FUEL_PRICE_NIS

                # --- טבלה בעברית ---
                results_df_display = results_df.copy()
                results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
                results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
                results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
                results_df_display = results_df_display.rename(columns=column_map_he)

                st.success(f"✅ התקבלו {len(results_df)} רכבים מגימניי.")
                st.dataframe(results_df_display.reset_index(drop=True))

                # דיסקליימר
                st.markdown("⚠️ **הבהרה חשובה**: הנתונים הם הערכה גסה של AI בלבד.", unsafe_allow_html=True)

                # --- הסברים בעברית ---
                st.markdown("### 📖 הסברים לכל פרמטר")
                for i, method in enumerate(methods_info, 1):
                    with st.expander(f"🔎 רכב {i} – הסברים"):
                        for k, v in method.items():
                            field_he = method_map_he.get(k, k)
                            st.write(f"- **{field_he}:** {v}")
            else:
                st.error("⚠️ לא נמצאו רכבים בפלט.")
