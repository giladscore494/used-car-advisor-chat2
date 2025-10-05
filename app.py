# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה מלאה עם גרף עלות כוללת והיצע בשוק (+ חישוב חשמל לרכבים חשמליים)
# =========================================

import streamlit as st
import pandas as pd
import json, os
from datetime import datetime
import numpy as np
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- Helpers --------
def init_state():
    for key in ["user_profile","validated_cars","methods_info","fuel_price","electricity_price"]:
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
        "main_use": main_use.strip(),  # תיאור חופשי (multiline), נשאר באנגלית/חופשי בשדה
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

# -------- Normalize values from Gemini --------
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
        df["turbo"] = df["turbo"].replace({
            "כן": True,
            "לא": False,
            True: True,
            False: False
        })
    return df

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
    "avg_fuel_consumption": "צריכת דלק ממוצעת (ק\"מ/ל')",  # יתעדכן דינאמית אם electric
    "annual_fee": "אגרה שנתית (₪)",
    "annual_energy_cost": "עלות אנרגיה שנתית (₪)",        # דינאמי דלק/חשמל
    "total_annual_cost": "עלות כוללת שנתית (₪)",
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
    "supply_method": "שיטת קביעת היצע"
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

# --- דלק וגיר ---
fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
if "חשמלי" in fuels_he:
    st.info("נבחר דלק חשמלי — תיבת ההילוכים נקבעת אוטומטית ל'אוטומטית'.")
    gears_he = ["אוטומטית"]
else:
    gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])

turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)

fuels = [fuel_map[f] for f in fuels_he]
gears = [gear_map[g] for g in gears_he]
turbo_choice = turbo_map[turbo_choice_he]

# --- פרטים אישיים ---
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

# --- היצע בשוק ---
consider_supply = st.radio("האם להתחשב בהיצע בשוק?", ["כן","לא"], index=0)

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
profile["consider_market_supply"] = (consider_supply == "כן")

# --- מחירי אנרגיה ---
fuel_price = st.number_input("מחיר ליטר דלק (₪)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
electricity_price = st.number_input("מחיר חשמל לקוט״ש (₪)", min_value=0.1, max_value=5.0, value=0.65, step=0.01)

st.session_state.fuel_price = fuel_price
st.session_state.electricity_price = electricity_price
profile["fuel_price_nis_per_liter"] = fuel_price
profile["electricity_price_nis_per_kwh"] = electricity_price

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
        1. החזר JSON יחיד עם שלושה שדות: "search_performed", "search_queries", "recommended_cars".
        2. search_performed: תמיד החזר True. עליך תמיד לבצע חיפוש אינטרנטי ולא להחזיר False.
        3. search_queries: החזר תמיד את מחרוזות החיפוש שבוצעו בפועל.
        4. recommended_cars: מערך של 5–10 רכבים. כל רכב חייב לכלול:
           - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
           - avg_fuel_consumption (לרכבים רגילים: ק\"מ/ל'; לרכבים חשמליים: קוט\"ש/100 ק\"מ, מספר בלבד) + fuel_method
           - annual_fee (₪ לשנה, מספר בלבד) + fee_method
           - reliability_score (מספר 1–10 בלבד) + reliability_method
           - maintenance_cost (₪ לשנה, מספר בלבד) + maintenance_method
           - safety_rating (מספר 1–10 בלבד) + safety_method
           - insurance_cost (₪ לשנה, מספר בלבד) + insurance_method
           - resale_value (מספר 1–10 בלבד) + resale_method
           - performance_score (מספר 1–10 בלבד) + performance_method
           - comfort_features (מספר 1–10 בלבד) + comfort_method
           - suitability (מספר 1–10 בלבד) + suitability_method
           - market_supply (\"גבוה\" / \"בינוני\" / \"נמוך\") + supply_method
        5. חובה להחזיר אך ורק מספרים עבור כל פרמטר ציון למעט שדה ההיצע.
        6. חובה להחזיר רכבים שנמכרים בפועל בישראל בלבד.
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
            search_performed = parsed.get("search_performed", False)
            search_queries = parsed.get("search_queries", [])

            if search_performed and search_queries:
                st.info("✅ בוצע חיפוש אינטרנטי לנתוני שוק עדכניים.")
            else:
                st.warning("⚠️ לא ברור אם בוצע חיפוש חי. ייתכן שהנתונים חלקיים.")

            cars_to_process = parsed["recommended_cars"]
            results_df, methods_info = clean_gemini_output(cars_to_process)

            if not results_df.empty:
                # --- Normalize Gemini values ---
                results_df = normalize_car_values(results_df)

                # --- חישוב עלות אנרגיה שנתית ---
                # הערה: avg_fuel_consumption מפורש כך:
                #  - לרכב חשמלי: קוט"ש/100 ק"מ
                #  - לרכב לא-חשמלי: ק"מ לליטר
                if "avg_fuel_consumption" not in results_df.columns:
                    st.error("חסר שדה avg_fuel_consumption בפלט.")
                    st.stop()

                # עזר לזיהוי חשמלי
                is_ev = results_df["fuel"].str.lower().eq("electric")

                # הגנות חלקיות מחלוקות ב-0
                km_per_liter = results_df["avg_fuel_consumption"].where(~is_ev, np.nan).replace(0, np.nan)
                kwh_per_100km = results_df["avg_fuel_consumption"].where(is_ev, np.nan)

                # עלות אנרגיה:
                annual_km = profile["annual_km"]
                fuel_price = st.session_state.fuel_price or 7.0
                elec_price = st.session_state.electricity_price or 0.65

                # דלק: (ק"מ / (ק"מ/ל')) * ₪/ל'
                fuel_cost = (annual_km / km_per_liter) * fuel_price
                # חשמל: (ק"מ / 100) * (קוט"ש/100ק"מ) * ₪/קוט"ש
                elec_cost = (annual_km / 100.0) * kwh_per_100km * elec_price

                results_df["annual_energy_cost"] = np.where(is_ev, elec_cost, fuel_cost)
                # לשמירת תאימות לאחור
                results_df["annual_fuel_cost"] = results_df["annual_energy_cost"]

                # --- עלות כוללת ---
                for col in ["maintenance_cost", "insurance_cost", "annual_fee"]:
                    if col not in results_df.columns:
                        results_df[col] = 0.0
                results_df["total_annual_cost"] = (
                    results_df["annual_energy_cost"].fillna(0) +
                    results_df["maintenance_cost"].fillna(0) +
                    results_df["insurance_cost"].fillna(0) +
                    results_df["annual_fee"].fillna(0)
                )

                # --- טבלה בעברית: כותרות דינמיות ---
                # צריכה: אם יש חשמליים → "צריכת חשמל (קוט\"ש/100 ק\"מ)" ; אחרת דלק
                if "fuel" in results_df.columns and results_df["fuel"].str.lower().eq("electric").any():
                    column_map_he["avg_fuel_consumption"] = "צריכת חשמל (קוט\"ש/100 ק\"מ)"
                    column_map_he["annual_energy_cost"] = "עלות חשמל שנתית (₪)"
                else:
                    column_map_he["avg_fuel_consumption"] = "צריכת דלק ממוצעת (ק\"מ/ל')"
                    column_map_he["annual_energy_cost"] = "עלות דלק שנתית (₪)"

                results_df_display = results_df.copy()
                results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
                results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
                results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
                results_df_display = results_df_display.rename(columns=column_map_he)

                st.success(f"✅ התקבלו {len(results_df)} רכבים מגימניי.")
                st.dataframe(results_df_display.reset_index(drop=True))

                # דיסקליימר
                st.markdown("⚠️ **הבהרה**: הנתונים הם הערכה גסה של AI; יש לאמת לפני קנייה.", unsafe_allow_html=True)

                # --- גרף השוואה ---
                st.markdown("### 📊 השוואת עלות כוללת שנתית")
                chart_df = results_df_display[["מותג", "דגם", "שנה", "עלות כוללת שנתית (₪)"]].copy()
                chart_df["רכב"] = chart_df["מותג"] + " " + chart_df["דגם"] + " " + chart_df["שנה"].astype(str)
                chart_df = chart_df.set_index("רכב")
                st.bar_chart(chart_df["עלות כוללת שנתית (₪)"])

                # --- הסברים בעברית ---
                st.markdown("### 📖 הסברים לכל פרמטר")
                for i, method in enumerate(methods_info, 1):
                    car_name = f"{results_df.iloc[i-1]['brand']} {results_df.iloc[i-1]['model']} {results_df.iloc[i-1]['year']}"
                    with st.expander(f"🔎 {car_name} – הסברים"):
                        for k, v in method.items():
                            field_he = method_map_he.get(k, k)
                            st.write(f"- **{field_he}:** {v}")
            else:
                st.error("⚠️ לא נמצאו רכבים בפלט.")