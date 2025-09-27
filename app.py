# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה עם FitScore + Search Meta Validation
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
                      reliability_weight, resale_weight, fuel_weight,
                      performance_weight, comfort_weight,
                      body_style, driving_style, excluded_colors):
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
        "weights": {
            "reliability": reliability_weight,
            "resale": resale_weight,
            "fuel": fuel_weight,
            "performance": performance_weight,
            "comfort": comfort_weight,
        },
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
    """מחשב ציון FitScore (עד 100) לפי המשקולות האישיות."""
    df['weighted_reliability']   = df['reliability_score'] * weights['reliability']
    df['weighted_resale']        = df['resale_value'] * weights['resale']
    df['weighted_performance']   = df['performance_score'] * weights['performance']
    df['weighted_comfort']       = df['comfort_features'] * weights['comfort']
    df['weighted_suitability']   = df['suitability'] * weights['fuel']

    df['FitScore'] = (
        df['weighted_reliability'] +
        df['weighted_resale'] +
        df['weighted_performance'] +
        df['weighted_comfort'] +
        df['weighted_suitability']
    )

    df['FitScore'] = round(df['FitScore'] / df['FitScore'].max() * 100, 1)
    return df.sort_values(by='FitScore', ascending=False)

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

st.markdown("#### נתוני משפחה ובטיחות")
family_size = st.selectbox("מספר נוסעים קבוע", ["1-2","3-4","5+"])
cargo_need = st.selectbox("דרישת תא מטען", ["קטן","בינוני","גדול"])
safety_required = st.selectbox("חובה מערכות בטיחות אקטיביות?", ["כן","לא"])
trim_level = st.selectbox("רמת אבזור", ["בסיסי","סטנדרטי","עשיר"])

st.markdown("#### סדר עדיפויות (1–5)")
reliability_weight = st.slider("אמינות", 1, 5, 5)
resale_weight = st.slider("שמירת ערך", 1, 5, 3)
fuel_weight = st.slider("חיסכון בדלק", 1, 5, 4)
performance_weight = st.slider("ביצועים", 1, 5, 3)
comfort_weight = st.slider("נוחות", 1, 5, 3)

st.markdown("#### העדפות נוספות")
body_style = st.selectbox("סגנון רכב מועדף", ["כל סוג","האצ'בק","סדאן","קרוסאובר/ג'יפון"])
driving_style = st.selectbox("אופי נהיגה עיקרי", ["רגוע ונינוח","דינמי וספורטיבי"])
excluded_colors = st.text_input("צבעים לפסילה (רשות)", value="")

profile = make_user_profile(
    budget_min, budget_max, [year_min, year_max],
    fuels, gears, turbo_choice, main_use, annual_km, driver_age,
    family_size, cargo_need, safety_required, trim_level,
    reliability_weight, resale_weight, fuel_weight,
    performance_weight, comfort_weight,
    body_style, driving_style, excluded_colors.split(",")
)
st.session_state.user_profile = profile

# -------- שלב 2: Gemini --------
st.markdown("### שלב 2: Gemini – המלצות ראשוניות")
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    if st.button("🚀 בקש המלצות מגימניי"):
        prompt = f"""
        אני צריך המלצות לרכבים מהשוק הישראלי בלבד. אלה הנתונים של הלקוח:
        {json.dumps(profile, ensure_ascii=False, indent=2)}

        דרישות לפלט:
        1. החזר אובייקט JSON יחיד עם שלושה שדות ברמה הראשית:
           - "search_performed": True/False
           - "search_queries": מערך מחרוזות עם כל שאילתות החיפוש שבוצעו
           - "recommended_cars": מערך של 5–10 רכבים
        2. כל רכב חייב לכלול: brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
        3. בנוסף: reliability_score, maintenance_cost, safety_rating,
           insurance_cost, resale_value, performance_score, comfort_features, suitability
           וכל אחד מהם עם שדה *_method שמסביר איך חושב.
        4. חובה להשתמש בנתוני שוק חיים ועדכניים כדי לקבוע price_range_nis ו-insurance_cost.
        5. החזר אך ורק JSON חוקי, ללא טקסט חיצוני.
        """

        with st.spinner("פונה לגימניי..."):
            try:
                resp = model.generate_content(prompt)
                text = resp.candidates[0].content.parts[0].text.strip()
                if text.startswith("```"):
                    text = text.strip("`").replace("json\n", "").replace("json", "").strip()

                try:
                    cars_from_gemini = json.loads(text)

                    if isinstance(cars_from_gemini, dict) and "recommended_cars" in cars_from_gemini:
                        search_performed = cars_from_gemini.get("search_performed", False)
                        search_queries = cars_from_gemini.get("search_queries", [])

                        if search_performed and search_queries:
                            st.info("✅ אימות נתונים: בוצע חיפוש אינטרנט לנתוני שוק עדכניים.")
                            st.code(search_queries)
                        else:
                            st.warning("⚠️ לא ברור אם בוצע חיפוש לנתונים עדכניים (סביר שהמודל השתמש בידע פנימי).")

                        cars_to_process = cars_from_gemini.get("recommended_cars", [])
                        if cars_to_process:
                            min_budget, max_budget = profile["budget_nis"]
                            results_df, methods_info = clean_gemini_output(cars_to_process, min_budget, max_budget)

                            if not results_df.empty:
                                ranked_df = calculate_fit_score(results_df.copy(), profile["weights"])
                                st.session_state.ranked_cars = ranked_df
                                st.session_state.methods_info = methods_info

                                st.success(f"✅ נמצאו {len(ranked_df)} רכבים אחרי סינון ודירוג.")
                                st.subheader("🏆 דירוג סופי (FitScore)")
                                st.dataframe(
                                    ranked_df.reset_index(drop=True).style.bar(
                                        subset=['FitScore'], color='#5cb85c'
                                    )
                                )

                                st.markdown("### 📖 נימוקים לכל רכב")
                                for i, (record, method) in enumerate(zip(ranked_df.to_dict(orient="records"), methods_info), 1):
                                    st.markdown(f"**🚘 {record.get('brand','')} {record.get('model','')} ({record.get('year','')}) — ⭐ {record.get('FitScore','N/A')}/100**")
                                    for k, v in method.items():
                                        st.write(f"- {k}: {v}")
                            else:
                                st.warning("⚠️ לא נמצאו רכבים מתאימים.")
                        else:
                            st.error("❌ recommended_cars ריק – אין רכבים לעיבוד.")
                    else:
                        st.error("⚠️ מבנה הפלט מגימניי אינו תקין. חסר 'recommended_cars'.")

                except json.JSONDecodeError:
                    st.error("⚠️ Gemini לא החזיר JSON חוקי. להלן הפלט:")
                    st.code(text)

            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
