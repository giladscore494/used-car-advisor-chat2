import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai

# =======================
# 🔑 API KEYS
# =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# =======================
# 📂 LOAD DATA
# =======================
@st.cache_data
def load_car_dataset():
    path = os.path.join(os.getcwd(), "car_models_israel_clean.csv")
    return pd.read_csv(path)

car_db = load_car_dataset()

# =======================
# 📖 BRAND DICTIONARY – 50 מותגים נפוצים בישראל
# =======================
BRAND_DICT = {
    "Toyota": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Hyundai": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Mazda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Kia": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Honda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Chevrolet": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Skoda": {"brand_country": "צ׳כיה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Ford": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Suzuki": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "סופר מיני"},
    "Seat": {"brand_country": "ספרד", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Volkswagen": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "משפחתי"},
    "Audi": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True, "category": "יוקרה"},
    "BMW": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "יוקרה"},
    "Mercedes": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True, "category": "יוקרה"},
    "Peugeot": {"brand_country": "צרפת", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True, "category": "משפחתי"},
    "Citroen": {"brand_country": "צרפת", "reliability": "נמוכה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Renault": {"brand_country": "צרפת", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Opel": {"brand_country": "גרמניה", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Mitsubishi": {"brand_country": "יפן", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True, "category": "משפחתי"},
    "Nissan": {"brand_country": "יפן", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Subaru": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Volvo": {"brand_country": "שוודיה", "reliability": "גבוהה", "demand": "בינוני", "luxury": True, "popular": False, "category": "יוקרה"},
    "Jeep": {"brand_country": "ארה״ב", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "SUV"},
    "Fiat": {"brand_country": "איטליה", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "עממי"},
}

# =======================
# 🧠 GPT – בחירת דגמים
# =======================
def ask_gpt_for_models(user_answers):
    for attempt in range(5):
        prompt = f"""
        בהתבסס על השאלון הבא, הצע עד 20 דגמים רלוונטיים בישראל.
        החזר JSON בלבד, בפורמט:
        [
          {{"model": "<string>", "year": <int>, "engine_cc": <int>, "fuel": "<string>", "gearbox": "<string>"}}
        ]

        שאלון:
        {json.dumps(user_answers, ensure_ascii=False)}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        st.text("==== RAW GPT RESPONSE ====")
        st.text(raw)
        try:
            return json.loads(raw)
        except Exception:
            continue
    return []

# =======================
# 🤖 GEMINI – השלמת נתונים
# =======================
def ask_gemini_for_specs(car_list, use_dict=True):
    for attempt in range(5):
        if use_dict:
            prompt = f"""
            החזר JSON עם המפתחות:
            - model
            - year
            - base_price_new
            - fuel_efficiency
            עבור הדגמים הבאים:
            {json.dumps(car_list, ensure_ascii=False)}
            """
        else:
            prompt = f"""
            החזר JSON עם המפתחות:
            - model
            - year
            - base_price_new
            - category
            - brand_country
            - reliability
            - demand
            - luxury
            - popular
            - fuel_efficiency
            עבור הדגמים הבאים:
            {json.dumps(car_list, ensure_ascii=False)}
            """

        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        st.text("==== RAW GEMINI RESPONSE ====")
        st.text(raw)
        try:
            return json.loads(raw)
        except Exception:
            continue
    return []

# =======================
# Normalize Gemini response
# =======================
def normalize_specs(raw_specs):
    specs_dict = {}
    if isinstance(raw_specs, list):
        for item in raw_specs:
            try:
                key = f"{item['model']} {item['year']}"
                specs_dict[key] = item
            except KeyError:
                continue
    elif isinstance(raw_specs, dict):
        specs_dict = raw_specs
    return specs_dict

# =======================
# 📉 נוסחת ירידת ערך
# =======================
def calculate_price(base_price_new, year, category, reliability, demand, fuel_efficiency):
    age = datetime.now().year - int(year)
    price = base_price_new
    price *= (1 - 0.07) ** age
    if category in ["מנהלים", "יוקרה"]:
        price *= 0.85
    elif category in ["מיני", "סופר מיני"]:
        price *= 0.95
    if reliability == "גבוהה":
        price *= 1.05
    elif reliability == "נמוכה":
        price *= 0.9
    if demand == "גבוה":
        price *= 1.05
    elif demand == "נמוך":
        price *= 0.9
    if fuel_efficiency >= 18:
        price *= 1.05
    elif fuel_efficiency <= 12:
        price *= 0.95
    if age > 10:
        price *= 0.85
    return round(price, -2)

# =======================
# 🔎 סינון
# =======================
def filter_results(cars, answers):
    filtered = []
    for car in cars:
        model_name = car["model"]
        calc_price = car["calculated_price"]
        if not (answers["budget_min"] * 0.87 <= calc_price <= answers["budget_max"] * 1.13):
            continue
        filtered.append(car)
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
    reliability_pref = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "חיסכון בדלק", "שמירת ערך"])
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
        "reliability_pref": reliability_pref,
    }

    st.info("📤 שולח בקשה ל־GPT לדגמים מתאימים...")
    gpt_models = ask_gpt_for_models(answers)

    final_cars = []
    dict_cars, fallback_cars = [], []

    for car in gpt_models:
        brand = car["model"].split()[0]
        if brand in BRAND_DICT:
            dict_cars.append(car)
        else:
            fallback_cars.append(car)

    if dict_cars:
        specs_dict_raw = ask_gemini_for_specs(dict_cars, use_dict=True)
        specs_dict = normalize_specs(specs_dict_raw)
        for car in dict_cars:
            brand = car["model"].split()[0]
            params = BRAND_DICT[brand]
            extra = specs_dict.get(f"{car['model']} {car['year']}", {})
            if not extra:
                st.warning(f"⚠️ לא נמצאו נתונים ל-{car['model']}")
                continue
            calc_price = calculate_price(
                extra["base_price_new"], car["year"],
                params["category"], params["reliability"], params["demand"],
                extra["fuel_efficiency"]
            )
            car["calculated_price"] = calc_price
            final_cars.append(car)

    if fallback_cars:
        specs_fb_raw = ask_gemini_for_specs(fallback_cars, use_dict=False)
        specs_fb = normalize_specs(specs_fb_raw)
        for car in fallback_cars:
            extra = specs_fb.get(f"{car['model']} {car['year']}", {})
            if not extra:
                st.warning(f"⚠️ לא נמצאו נתונים ל-{car['model']}")
                continue
            calc_price = calculate_price(
                extra["base_price_new"], car["year"],
                extra["category"], extra["reliability"], extra["demand"],
                extra["fuel_efficiency"]
            )
            car["calculated_price"] = calc_price
            final_cars.append(car)

    filtered = filter_results(final_cars, answers)

    if filtered:
        st.success("✅ נמצאו רכבים מתאימים:")
        st.dataframe(pd.DataFrame(filtered))
    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")