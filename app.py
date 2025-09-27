# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה עם 16 פרמטרים והסברים, מוגבל לשוק הישראלי
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

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
    }

# ניקוי פלט Gemini
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

profile = make_user_profile(budget_min, budget_max, [year_min, year_max],
                            fuels, gears, turbo_choice, main_use, annual_km, driver_age)
st.session_state.user_profile = profile
st.json(profile)

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
                resp = model.generate_content(prompt)
                text = resp.candidates[0].content.parts[0].text.strip()

                if text.startswith("```"):
                    text = text.strip("`")
                    text = text.replace("json\n", "").replace("json", "").strip()

                try:
                    cars_from_gemini = json.loads(text)
                    st.subheader("📋 פלט ראשוני מגימניי")
                    st.dataframe(pd.DataFrame(cars_from_gemini))
                except json.JSONDecodeError:
                    st.error("⚠️ גימניי לא החזיר JSON טהור. להלן הפלט:")
                    st.code(text)
                    cars_from_gemini = []

            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
                cars_from_gemini = []

        # ✅ ניקוי וסינון לפי תקציב
        if cars_from_gemini:
            min_budget, max_budget = profile["budget_nis"]
            results_df, methods_info = clean_gemini_output(cars_from_gemini, min_budget, max_budget)

            if not results_df.empty:
                st.session_state.validated_cars = results_df
                st.session_state.methods_info = methods_info

                st.success(f"✅ נמצאו {len(results_df)} רכבים אחרי סינון לפי תקציב (שוק ישראלי בלבד).")
                st.dataframe(results_df.reset_index(drop=True))

                # הצגת ההסברים
                st.markdown("### 📖 הסברים לכל פרמטר")
                for i, method in enumerate(methods_info, 1):
                    st.markdown(f"**רכב {i}:**")
                    for k, v in method.items():
                        st.write(f"- {k}: {v}")
            else:
                st.warning("⚠️ לא נמצאו רכבים שעומדים בתקציב.")
