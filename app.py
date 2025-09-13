import os
import re
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import requests

# =======================
# 🔑 API KEYS
# =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =======================
# 📂 LOAD DATA
# =======================
@st.cache_data
def load_car_dataset():
    path = os.path.join(os.getcwd(), "car_models_israel_clean.csv")
    return pd.read_csv(path)

car_db = load_car_dataset()

# =======================
# 🗂️ BRAND DICTIONARY + TRANSLATION
# =======================
BRAND_DICT = {
    "Toyota": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Hyundai": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Mazda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Kia": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Honda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Ford": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Volkswagen": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "משפחתי"},
    "Audi": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True, "category": "יוקרה"},
    "BMW": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "יוקרה"},
    "Mercedes": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True, "category": "יוקרה"},
    "Suzuki": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "סופר מיני"},
}

BRAND_TRANSLATION = {
    "יונדאי": "Hyundai",
    "מאזדה": "Mazda",
    "טויוטה": "Toyota",
    "קיה": "Kia",
    "הונדה": "Honda",
    "פורד": "Ford",
    "פולקסווגן": "Volkswagen",
    "אודי": "Audi",
    "ב.מ.וו": "BMW",
    "מרצדס": "Mercedes",
    "סוזוקי": "Suzuki",
}

# =======================
# 🧮 נוסחת ירידת ערך
# =======================
def calculate_price(base_price_new, year, category, brand_country,
                    reliability, demand, popular, fuel_efficiency):
    current_year = datetime.now().year
    age = current_year - year

    if age <= 5:
        depreciation_rate = 0.10
    elif age <= 10:
        depreciation_rate = 0.15
    else:
        depreciation_rate = 0.22

    if category in ["יוקרה", "מנהלים"] or brand_country in ["גרמניה", "ארה״ב"]:
        depreciation_rate += 0.03
    elif brand_country in ["יפן", "קוריאה"]:
        depreciation_rate -= 0.02

    if demand == "גבוה":
        depreciation_rate -= 0.02
    elif demand == "נמוך":
        depreciation_rate += 0.02

    if reliability == "גבוהה":
        depreciation_rate -= 0.02
    elif reliability == "נמוכה":
        depreciation_rate += 0.03

    price_est = base_price_new * ((1 - depreciation_rate) ** age)
    price_est = max(price_est, 5000)

    price_low = int(price_est * 0.9)
    price_high = int(price_est * 1.1)

    return price_low, int(price_est), price_high

# =======================
# 🔎 סינון
# =======================
def filter_results(cars, answers):
    filtered = []
    dropped_price, dropped_turbo = [], []

    for car in cars:
        calc_low = car.get("price_low")
        calc_est = car.get("price_est")
        calc_high = car.get("price_high")

        if calc_est is None:
            continue

        # 🔎 סינון תקציב
        if not (answers["budget_min"] <= calc_high and answers["budget_max"] >= calc_low):
            dropped_price.append(car)
            continue

        # 🔎 סינון טורבו
        if answers["turbo"] != "לא משנה":
            turbo_required = True if answers["turbo"] == "כן" else False
            if car.get("turbo") != turbo_required:
                dropped_turbo.append(car)
                continue

        filtered.append(car)

    # 📋 Debug
    if dropped_price:
        st.warning("❌ רכבים שנפלו בגלל תקציב:")
        st.dataframe(pd.DataFrame(dropped_price))
    if dropped_turbo:
        st.warning("❌ רכבים שנפלו בגלל טורבו:")
        st.dataframe(pd.DataFrame(dropped_turbo))

    return filtered

# =======================
# 🎛️ STREAMLIT APP
# =======================
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

with st.form("car_form"):
    budget_min = st.number_input("תקציב מינימלי (₪)", value=20000)
    budget_max = st.number_input("תקציב מקסימלי (₪)", value=40000)
    engine_min = st.number_input("נפח מנוע מינימלי (סמ״ק)", value=1200)
    engine_max = st.number_input("נפח מנוע מקסימלי (סמ״ק)", value=1800)
    year_min = st.number_input("שנת ייצור מינימלית", value=2010)
    year_max = st.number_input("שנת ייצור מקסימלית", value=2020)
    fuel = st.selectbox("מנוע מועדף", ["בנזין", "דיזל", "היברידי", "חשמלי"])
    gearbox = st.selectbox("גיר", ["לא משנה", "אוטומט", "ידני"])
    body_type = st.text_input("סוג רכב (למשל: סדאן, SUV, האצ׳בק)")
    turbo = st.selectbox("מנוע טורבו", ["לא משנה", "כן", "לא"])
    reliability_pref = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "חיסכון בדלק", "שמירת ערך"])
    extra_notes = st.text_area("הערות חופשיות (אופציונלי)", "")
    submit = st.form_submit_button("מצא רכבים")

if submit:
    answers = {
        "budget_min": budget_min,
        "budget_max": budget_max,
        "engine_min": engine_min,
        "engine_max": engine_max,
        "year_min": year_min,
        "year_max": year_max,
        "fuel": fuel,
        "gearbox": gearbox,
        "body_type": body_type,
        "turbo": turbo,
        "reliability_pref": reliability_pref,
        "extra_notes": extra_notes,
    }

    st.info("⚙️ Debug: תשובות משתמש")
    st.json(answers)

    # כאן ממשיך החיבור ל-GPT ול-Perplexity + חישוב מחירים