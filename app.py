# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה מלאה עם חיפוש חי, FitScore ודירוג סופי
# =========================================

import streamlit as st
import pandas as pd
import json, os
from datetime import datetime
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- Helpers --------
def init_state():
    for key in ["user_profile","validated_cars","methods_info","ranked_cars"]:
        if key not in st.session_state:
            st.session_state[key] = None

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age,
                      family_size, cargo_need, safety_required, trim_level,
                      weights, body_style, driving_style, excluded_colors):
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

def clean_gemini_output(cars_raw, min_budget, max_budget):
    records, methods = [], []
    for car in cars_raw:
        if not isinstance(car, dict):
            continue

        # סינון לפי תקציב
        price_min, price_max = None, None
        if isinstance(car.get("price_range_nis"), list) and len(car["price_range_nis"]) == 2:
            try:
                price_min, price_max = map(int, car["price_range_nis"])
            except Exception:
                pass

        price_ok = True
        if price_min and price_max:
            price_ok = (price_min >= min_budget and price_max <= max_budget)
        if not price_ok:
            continue

        # פיצול ערכים ומטודות
        record, method = {}, {}
        for k, v in car.items():
            if k.endswith("_method"):
                method[k] = v
            else:
                record[k] = v

        records.append(record)
        methods.append(method)

    return pd.DataFrame(records), methods

def calculate_fit_score(df, weights):
    """מחשב ציון התאמה (FitScore) לכל רכב על בסיס ציוני Gemini ומשקולות המשתמש."""

    df['weighted_reliability']   = df['reliability_score']   * weights['reliability']
    df['weighted_resale']        = df['resale_value']        * weights['resale']
    df['weighted_performance']   = df['performance_score']   * weights['performance']
    df['weighted_comfort']       = df['comfort_features']    * weights['comfort']
    df['weighted_suitability']   = df['suitability']         * weights['fuel']   # שימוש בציון התאמה כחיסכון

    df['FitScore'] = (
        df['weighted_reliability'] +
        df['weighted_resale'] +
        df['weighted_performance'] +
        df['weighted_comfort'] +
        df['weighted_suitability']
    )

    # נרמול ל־100
    max_score = (10 * sum(weights.values()))
    df['FitScore'] = round(df['FitScore'] / max_score * 100, 1)

    df = df.sort_values(by='FitScore', ascending=False)
    return df

# -------- עיצוב כרטיסים --------
st.markdown("""
<style>
.car-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
.car-card h3 {
    margin-top: 0;
    margin-bottom: 12px;
    color: #222222;
}
.car-card ul {
    padding-left: 0;
    margin: 0;
    list-style-type: none;
}
.car-card li {
    margin-bottom: 8px;
    font-size: 15px;
    display: flex;
    align-items: center;
}
.label {
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: bold;
    color: white;
    margin-right: 8px;
    font-size: 13px;
}
.label-reliability { background-color: #4caf50; }
.label-maintenance { background-color: #ff9800; }
.label-safety { background-color: #f44336; }
.label-insurance { background-color: #2196f3; }
.label-resale { background-color: #9c27b0; }
.label-performance { background-color: #ffc107; color:#000; }
.label-comfort { background-color: #e91e63; }
.label-suitability { background-color: #009688; }
</style>
""", unsafe_allow_html=True)

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

# שאלות נוספות
family_size = st.selectbox("מספר נוסעים קבוע", ["1-2","3-4","5+"])
cargo_need = st.selectbox("נפח מטען", ["קטן","בינוני","גדול"])
safety_required = st.selectbox("מערכות בטיחות אקטיביות חובה?", ["כן","לא"])
trim_level = st.selectbox("רמת אבזור פנימי", ["בסיסי","סטנדרטי","עשיר"])
body_style = st.selectbox("סגנון גוף מועדף", ["כל סוג","קרוסאובר","סדאן","האצ'בק"])
driving_style = st.selectbox("אופי הנהיגה", ["רגוע ונינוח","דינמי וספורטיבי"])
excluded_colors = st.text_input("צבעים לפסילה (רשימה מופרדת בפסיקים)", value="").split(",")

# סדר עדיפויות (משקולות)
st.markdown("#### סדר עדיפויות (1–5)")
c7,c8,c9,c10,c11 = st.columns(5)
with c7: reliability_weight = st.slider("אמינות",1,5,5)
with c8: resale_weight = st.slider("שמירת ערך",1,5,3)
with c9: fuel_weight = st.slider("חיסכון בדלק",1,5,4)
with c10: performance_weight = st.slider("ביצועים",1,5,3)
with c11: comfort_weight = st.slider("נוחות",1,5,2)

weights = {
    "reliability": reliability_weight,
    "resale": resale_weight,
    "fuel": fuel_weight,
    "performance": performance_weight,
    "comfort": comfort_weight
}

profile = make_user_profile(budget_min, budget_max, [year_min, year_max],
                            fuels, gears, turbo_choice, main_use, annual_km, driver_age,
                            family_size, cargo_need, safety_required, trim_level,
                            weights, body_style, driving_style, excluded_colors)
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
        אני צריך המלצות לרכבים. אלה התכונות שהלקוח חיפש:
        {json.dumps(profile, ensure_ascii=False, indent=2)}

        דרישות לפלט:
        1. החזר מערך JSON בלבד (ללא טקסט חיצוני).
        2. כל רכב חייב לכלול בדיוק את 16 הפרמטרים הבאים + שדה הסבר נלווה:
           - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
           - reliability_score (1–10) + reliability_method
           - maintenance_cost (₪ לשנה) + maintenance_method
           - safety_rating (1–10) + safety_method
           - insurance_cost (₪ לשנה) + insurance_method
           - resale_value (1–10) + resale_method
           - performance_score (1–10) + performance_method
           - comfort_features (1–10) + comfort_method
           - suitability (1–10) + suitability_method
        3. כל שדה *_method יסביר בקצרה איך חושב הערך.
        4. החזר 5–10 רכבים בלבד.
        5. חובה להחזיר רכבים שנמכרים בפועל בישראל (שוק הרכב הישראלי בלבד).
        6. אל תחזיר רכבים שלא זמינים בארץ.
        7. חובה לעמוד בטווח התקציב והשנתון שהמשתמש הזין.
        """

        with st.spinner("פונה לגימניי..."):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json"
                    },
                    tools=[{"google_search": {}}]
                )
                text = resp.candidates[0].content.parts[0].text.strip()

                cars_from_gemini = json.loads(text)

                st.subheader("📋 פלט ראשוני מגימניי")
                st.dataframe(pd.DataFrame(cars_from_gemini))

            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
                cars_from_gemini = []

        # ✅ ניקוי, סינון וחישוב FitScore
        if cars_from_gemini:
            min_budget, max_budget = profile["budget_nis"]
            results_df, methods_info = clean_gemini_output(cars_from_gemini, min_budget, max_budget)

            if not results_df.empty:
                st.session_state.validated_cars = results_df
                st.session_state.methods_info = methods_info

                # שלב 3 – חישוב FitScore
                ranked_df = calculate_fit_score(results_df.copy(), profile["weights"])
                st.session_state.ranked_cars = ranked_df

                st.subheader("🏆 שלב 3: דירוג סופי (FitScore)")
                st.markdown("הרכבים מדורגים לפי סדרי העדיפויות האישיים שלך:")
                st.dataframe(ranked_df.reset_index(drop=True).style.bar(
                    subset=['FitScore'], color='#5cb85c'
                ))

                # הצגת ההסברים
                st.markdown("## 📖 נימוקים לכל רכב")
                icons = {
                    "reliability_method": ("🛡️ אמינות", "label-reliability"),
                    "maintenance_method": ("🔧 תחזוקה", "label-maintenance"),
                    "safety_method": ("🧯 בטיחות", "label-safety"),
                    "insurance_method": ("💰 ביטוח", "label-insurance"),
                    "resale_method": ("📉 שמירת ערך", "label-resale"),
                    "performance_method": ("⚡ ביצועים", "label-performance"),
                    "comfort_method": ("🛋️ נוחות", "label-comfort"),
                    "suitability_method": ("🎯 התאמה כוללת", "label-suitability"),
                }

                for i, (record, method) in enumerate(zip(ranked_df.to_dict(orient="records"), methods_info), 1):
                    car_title = f"🚘 {record.get('brand','')} {record.get('model','')} ({record.get('year','')})"
                    fit_score = record.get("FitScore", "N/A")

                    explanations = "<ul>"
                    for key, (label, css_class) in icons.items():
                        if key in method:
                            explanations += f"""
                            <li>
                                <span class="label {css_class}">{label}</span> {method[key]}
                            </li>
                            """
                    explanations += "</ul>"

                    st.markdown(f"""
                    <div class="car-card">
                        <h3>{car_title} — ⭐ {fit_score}/100</h3>
                        {explanations}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ לא נמצאו רכבים שעומדים בתקציב.")
