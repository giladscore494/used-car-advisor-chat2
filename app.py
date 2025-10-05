# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה מלאה עם חישוב חשמל אוטומטי וטעינה ביתית/ציבורית
# =========================================

import streamlit as st
import pandas as pd
import json, os
import numpy as np
from datetime import datetime, date
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- Helpers --------
def init_state():
    for key in ["user_profile","validated_cars","methods_info","fuel_price"]:
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
            "ידני": "manual",
            "ידנית": "manual"
        })
    if "turbo" in df.columns:
        df["turbo"] = df["turbo"].replace({"כן": True, "לא": False})
    return df

# -------- מיפויים --------
fuel_map = {"בנזין": "gasoline", "היברידי": "hybrid", "דיזל": "diesel", "חשמלי": "electric"}
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
    "annual_energy_cost": "עלות אנרגיה שנתית (₪)",
    "annual_home_charge": "טעינה ביתית שנתית (₪)",
    "annual_public_charge": "טעינה ציבורית שנתית (₪)",
    "total_annual_cost": "עלות כוללת שנתית (₪)",
    "range_estimate": "טווח נסיעה משוער (ק\"מ)",
    "reliability_score": "אמינות",
    "maintenance_cost": "עלות אחזקה (₪/שנה)",
    "safety_rating": "בטיחות",
    "insurance_cost": "עלות ביטוח (₪/שנה)",
    "resale_value": "שמירת ערך",
    "performance_score": "ביצועים",
    "comfort_features": "נוחות",
    "suitability": "התאמה",
    "market_supply": "היצע בשוק"
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

# גיר אוטומטי לרכב חשמלי
if "חשמלי" in fuels_he:
    st.info("נבחר רכב חשמלי – תיבת ההילוכים נקבעת לאוטומטית בלבד.")
    gears_he = ["אוטומטית"]
else:
    gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])

turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)
fuels = [fuel_map[f] for f in fuels_he]
gears = [gear_map[g] for g in gears_he]
turbo_choice = turbo_map[turbo_choice_he]

c4, c5, c6 = st.columns([2,1,1])
with c4:
    main_use = st.text_area("תיאור הרכב והשימוש בו", value="נסיעה יומיומית לעבודה וטיולים קצרים", height=100)
with c5:
    annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
with c6:
    driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)

license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])
insurance_history = st.text_input("עבר ביטוחי", value="שנתיים ללא תביעות")
violations = st.selectbox("דוחות/שלילות", ["אין", "שלילה בעבר", "נקודות פעילות"])
family_size = st.selectbox("גודל משפחה", ["1-2","3-4","5+"])
cargo_need = st.selectbox("צורך בתא מטען", ["קטן","בינוני","גדול"])
safety_required = st.radio("חובה מערכות בטיחות אקטיביות?", ["כן","לא"])
trim_level = st.selectbox("רמת אבזור", ["בסיסי","סטנדרטי","עשיר"])
consider_supply = st.radio("האם להתחשב בהיצע בשוק?", ["כן","לא"], index=0)

# --- מחירי אנרגיה אוטומטיים ---
is_electric = "חשמלי" in fuels_he
today_str = date.today().strftime("%d.%m.%Y")

# תעריפי חשמל מעודכנים בישראל (אוקטובר 2025)
current_electricity_price_home = 0.67
current_electricity_price_public = 1.55

if is_electric:
    st.info("⚡ נבחר רכב חשמלי – יוצגו מחירי טעינה לפי התעריפים המעודכנים בישראל.")
    fuel_price = None
    electricity_price_home = current_electricity_price_home
    electricity_price_public = current_electricity_price_public
else:
    col_fuel, col_note = st.columns([2,1])
    with col_fuel:
        fuel_price = st.number_input(
            "מחיר ליטר דלק (₪)",
            min_value=1.0, max_value=20.0,
            value=7.0, step=0.1
        )
    with col_note:
        st.markdown(
            f"💡 **עלות חשמל מעודכנת ל־{today_str}:** "
            f"{current_electricity_price_home:.2f}₪ לקוט״ש (ביתית), "
            f"{current_electricity_price_public:.2f}₪ לקוט״ש (ציבורית)"
        )
    electricity_price_home = current_electricity_price_home
    electricity_price_public = current_electricity_price_public

# --- משקולות ופרופיל ---
weights = {
    "reliability": st.slider("אמינות", 1, 5, 5),
    "resale": st.slider("שמירת ערך", 1, 5, 3),
    "fuel": st.slider("חיסכון בדלק", 1, 5, 4),
    "performance": st.slider("ביצועים", 1, 5, 2),
    "comfort": st.slider("נוחות", 1, 5, 3),
}
body_style = st.selectbox("סגנון מרכב מועדף", ["כללי","סדאן","האצ'בק","קרוסאובר/ג'יפון"])
driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח","דינמי וספורטיבי"])
excluded_colors = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="").split(",")

profile = make_user_profile(
    budget_min, budget_max, [year_min, year_max],
    fuels, gears, turbo_choice, main_use, annual_km, driver_age,
    family_size, cargo_need, safety_required, trim_level,
    weights, body_style, driving_style, excluded_colors
)
profile.update({
    "license_years": license_years,
    "driver_gender": driver_gender,
    "insurance_history": insurance_history,
    "violations": violations,
    "consider_market_supply": (consider_supply == "כן"),
    "fuel_price_nis_per_liter": fuel_price,
    "electricity_home_price": electricity_price_home,
    "electricity_public_price": electricity_price_public
})
st.session_state.user_profile = profile

# -------- שלב 2 --------
st.markdown("### שלב 2: Gemini – המלצות ראשוניות")
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    if st.button("🚀 בקש המלצות מגימניי"):
        with st.spinner("פונה לגימניי..."):
            try:
                resp = model.generate_content(f"""
                אני צריך המלצות לרכבים ללקוח ישראלי. זה הפרופיל:
                {json.dumps(profile, ensure_ascii=False, indent=2)}
                """)
                text = resp.candidates[0].content.parts[0].text.strip().replace("```json","").replace("```","").strip()
                parsed = json.loads(text)
            except Exception as e:
                st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
                parsed = {}

        if parsed and "recommended_cars" in parsed:
            df, methods = clean_gemini_output(parsed["recommended_cars"])
            if not df.empty:
                df = normalize_car_values(df)
                is_ev = df["fuel"].str.lower().eq("electric")

                annual_km = profile["annual_km"]
                fuel_p = profile.get("fuel_price_nis_per_liter", 7.0)
                elec_home = profile.get("electricity_home_price", 0.67)
                elec_public = profile.get("electricity_public_price", 1.55)

                km_per_l = df["avg_fuel_consumption"].where(~is_ev, np.nan)
                kwh_per_100km = df["avg_fuel_consumption"].where(is_ev, np.nan)

                df["annual_home_charge"] = (annual_km / 100) * kwh_per_100km * elec_home
                df["annual_public_charge"] = (annual_km / 100) * kwh_per_100km * elec_public
                fuel_cost = (annual_km / km_per_l) * fuel_p
                df["annual_energy_cost"] = np.where(is_ev, df["annual_home_charge"], fuel_cost)

                # טווח נסיעה משוער (בק״מ)
                df["range_estimate"] = np.where(
                    is_ev, (100 / kwh_per_100km * 60).round(0), (km_per_l * 45).round(0)
                )

                df["total_annual_cost"] = (
                    df["annual_energy_cost"].fillna(0)
                    + df["maintenance_cost"].fillna(0)
                    + df["insurance_cost"].fillna(0)
                    + df["annual_fee"].fillna(0)
                )

                df_display = df.rename(columns=column_map_he)
                df_display["fuel"] = df_display["fuel"].map(fuel_map_he).fillna(df_display["fuel"])

                st.success(f"✅ התקבלו {len(df)} רכבים מגימניי.")
                st.dataframe(df_display)

                st.markdown("💡 **תעריפי חשמל מעודכנים ל־{}:** {:.2f}₪ (ביתית), {:.2f}₪ (ציבורית)".format(
                    today_str, current_electricity_price_home, current_electricity_price_public
                ))

                st.markdown("### 📊 השוואת עלות כוללת שנתית")
                chart_df = df_display[["מותג","דגם","שנה","עלות כוללת שנתית (₪)"]].copy()
                chart_df["רכב"] = chart_df["מותג"] + " " + chart_df["דגם"] + " " + chart_df["שנה"].astype(str)
                chart_df = chart_df.set_index("רכב")
                st.bar_chart(chart_df["עלות כוללת שנתית (₪)"])