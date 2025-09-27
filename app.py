# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה עם טעינה אוטומטית + Fuzzy Matching מותאם למבנה הקובץ שלך
# =========================================

import streamlit as st
import pandas as pd
import json, os
from datetime import datetime
import google.generativeai as genai
from rapidfuzz import fuzz

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- Helpers --------
def init_state():
    for key in ["inventory_df","user_profile","validated_cars"]:
        if key not in st.session_state:
            st.session_state[key] = None

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],   # נשמר, אבל לא מסונן בפועל (אין מחירים בקובץ)
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
    }

# -------- שלב 1: שאלון + מאגר --------
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

fuels = st.multiselect("סוגי דלק מועדפים", ["gasoline","hybrid","diesel","electric"], default=["gasoline"])
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

# --- טעינה אוטומטית של המאגר ---
st.markdown("### שלב 1ב: טעינת מאגר (אוטומטי)")
default_path = "car_models_israel_clean.csv"
if os.path.exists(default_path):
    df = pd.read_csv(default_path, encoding="utf-8-sig")

    # מיפוי דלקים לעקביות באנגלית
    fuel_map = {"בנזין":"gasoline","דיזל":"diesel","היברידי-בנזין":"hybrid","חשמלי":"electric"}
    if df["fuel"].dtype == "object":
        df["fuel"] = df["fuel"].map(fuel_map).fillna(df["fuel"])

    # המרה לערך קריא של הילוכים
    if df["automatic"].dtype in ["int64","float64"]:
        df["automatic"] = df["automatic"].apply(lambda x: "automatic" if x==1 else "manual")

    st.session_state.inventory_df = df
    st.success(f"מאגר ברירת מחדל נטען ({len(df)} שורות, {df['brand'].nunique()} מותגים).")
else:
    st.error("❌ לא נמצא קובץ car_models_israel_clean.csv בתיקייה!")

# -------- שלב 2: Gemini + סינון --------
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

        שלבים:
        1. חשוב לפי הנתונים בשאלון.
        2. בצע חיפוש ברשת למחירים עדכניים ולזמינות הדגמים בישראל.
        3. דרג לפי חיסכון, אמינות, עלויות תחזוקה.
        4. החזר 5–10 רכבים בלבד.
        5. החזר אך ורק בפורמט JSON תקין, בלי טקסט נוסף.
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
                    st.subheader("📋 פלט ראשוני מגימניי (לפני סינון)")
                    st.dataframe(pd.DataFrame(cars_from_gemini))
                except json.JSONDecodeError:
                    st.error("⚠️ גימניי לא החזיר JSON טהור. להלן הפלט:")
                    st.code(text)
                    cars_from_gemini = []

            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
                cars_from_gemini = []

        valid_cars = []
        if st.session_state.inventory_df is not None and cars_from_gemini:
            df_inv = st.session_state.inventory_df

            for car in cars_from_gemini:
                found_match = False
                for _, row in df_inv.iterrows():
                    brand_sim = fuzz.ratio(str(car.get("brand","")).lower(), str(row["brand"]).lower())
                    model_sim = fuzz.partial_ratio(str(car.get("model","")).lower(), str(row["model"]).lower())
                    year_match = ("year" in car and row["year"] == car["year"])
                    if brand_sim >= 85 and model_sim >= 80 and year_match:
                        found_match = True
                        break

                if found_match:
                    valid_cars.append(car)

            st.session_state.validated_cars = pd.DataFrame(valid_cars)
            if not st.session_state.validated_cars.empty:
                st.success(f"✅ נמצאו {len(st.session_state.validated_cars)} רכבים אחרי סינון.")
                st.dataframe(st.session_state.validated_cars)
            else:
                st.warning("⚠️ לא נמצאו רכבים שעוברים את הסינון מול המאגר.")
