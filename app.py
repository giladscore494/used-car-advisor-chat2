# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה עם נרמול, ניקוי פלט Gemini והצגת ציוני התאמה
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
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
    }

# פונקציות נרמול
def normalize_fuel(fuel: str) -> str:
    mapping = {
        "gasoline": "gasoline",
        "בנזין": "gasoline",
        "diesel": "diesel",
        "דיזל": "diesel",
        "hybrid": "hybrid",
        "היברידי-בנזין": "hybrid",
        "hybrid-diesel": "hybrid-diesel",
        "היברידי-דיזל": "hybrid-diesel",
        "electric": "electric",
        "חשמלי": "electric"
    }
    return mapping.get(str(fuel).lower().strip(), fuel)

def normalize_gear(gear: str) -> str:
    mapping = {
        "automatic": "automatic",
        "אוטומט": "automatic",
        "ידני": "manual",
        "manual": "manual",
        "1": "automatic",
        "0": "manual"
    }
    return mapping.get(str(gear).lower().strip(), gear)

def flexible_match(a, b):
    if not a or not b:
        return 0
    return max(
        fuzz.ratio(str(a).lower(), str(b).lower()),
        fuzz.partial_ratio(str(a).lower(), str(b).lower()),
        fuzz.token_sort_ratio(str(a).lower(), str(b).lower())
    )

# ניקוי פלט Gemini
def clean_gemini_output(cars_raw, df_inv, min_budget, max_budget):
    cleaned = []
    for car in cars_raw:
        if not isinstance(car, dict):
            continue

        brand = str(car.get("brand","")).strip()
        model = str(car.get("model","")).strip()
        year = int(car.get("year",0)) if str(car.get("year","")).isdigit() else None
        fuel = normalize_fuel(car.get("fuel",""))
        gear = normalize_gear(car.get("gear",""))

        # מחיר (טווח)
        price_min, price_max = None, None
        if isinstance(car.get("price_range_nis"), list) and len(car["price_range_nis"]) == 2:
            try:
                price_min, price_max = map(int, car["price_range_nis"])
            except Exception:
                pass

        # השוואה מול המאגר
        best_brand_sim, best_model_sim = 0, 0
        found_match = False
        for _, row in df_inv.iterrows():
            brand_sim = flexible_match(brand, row["brand"])
            model_sim = flexible_match(model, row["model"])
            year_match = (year and row["year"] == year)
            fuel_match = (not fuel or fuel == normalize_fuel(row["fuel"]))
            gear_match = (not gear or gear == normalize_gear(row["automatic"]))

            if brand_sim > best_brand_sim: best_brand_sim = brand_sim
            if model_sim > best_model_sim: best_model_sim = model_sim

            if brand_sim >= 70 and model_sim >= 65 and year_match and fuel_match and gear_match:
                found_match = True
                break

        # בדיקת תקציב
        price_ok = True
        if price_min and price_max:
            price_ok = (price_min >= min_budget and price_max <= max_budget)

        if found_match and price_ok:
            cleaned.append({
                "brand": brand,
                "model": model,
                "year": year,
                "fuel": fuel,
                "gear": gear,
                "price_min": price_min,
                "price_max": price_max,
                "brand_sim": best_brand_sim,
                "model_sim": best_model_sim
            })

    return pd.DataFrame(cleaned)

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

# --- טעינה אוטומטית של המאגר ---
st.markdown("### שלב 1ב: טעינת מאגר (אוטומטי)")
default_path = "car_models_israel_clean.csv"
if os.path.exists(default_path):
    df = pd.read_csv(default_path, encoding="utf-8-sig")

    # נרמול דלק והילוכים
    if df["fuel"].dtype == "object":
        df["fuel"] = df["fuel"].map(normalize_fuel).fillna(df["fuel"])
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
        5. חובה להחזיר בכל רכב שדות:
           - brand (באנגלית בלבד)
           - model (באנגלית בלבד)
           - year (מספר)
           - fuel (gasoline/diesel/hybrid/hybrid-diesel/electric)
           - gear (automatic/manual)
           - price_range_nis (מערך עם שני מספרים)
        6. החזר אך ורק בפורמט JSON תקין, בלי טקסט נוסף.
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

        if st.session_state.inventory_df is not None and cars_from_gemini:
            df_inv = st.session_state.inventory_df
            min_budget, max_budget = profile["budget_nis"]

            results_df = clean_gemini_output(cars_from_gemini, df_inv, min_budget, max_budget)

            if not results_df.empty:
                st.success(f"✅ נמצאו {len(results_df)} רכבים אחרי סינון וניקוי.")
                st.dataframe(results_df.sort_values(by=["brand_sim","model_sim"], ascending=False).reset_index(drop=True))
            else:
                st.warning("⚠️ לא נמצאו רכבים שעוברים את הסינון מול המאגר והתקציב.")
