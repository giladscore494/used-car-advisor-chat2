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
# 🧠 GPT – בחירת דגמים
# =======================
def ask_gpt_for_models(user_answers, max_retries=5):
    prompt = f"""
    בהתבסס על השאלון הבא, הצע עד 20 דגמים רלוונטיים בישראל.
    החזר JSON בלבד, בפורמט:
    [
      {{
        "model": "<string>",
        "year": <int>,
        "engine_cc": <int>,
        "fuel": "<string>",
        "gearbox": "<string>",
        "turbo": <true/false>
      }}
    ]

    שאלון:
    {json.dumps(user_answers, ensure_ascii=False)}
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            raw = response.choices[0].message.content.strip()
            st.text_area(f"==== RAW GPT RESPONSE (attempt {attempt+1}) ====", raw, height=200)

            if raw.startswith("```"):
                raw = raw.strip("```json").strip("```").strip()

            models = json.loads(raw)
            return models
        except Exception as e:
            st.warning(f"⚠️ GPT ניסיון {attempt+1} נכשל: {e}")
    return []

# =======================
# 🌐 PERPLEXITY – השלמת נתוני רכב (בבת אחת)
# =======================
def parse_kv_format_block(text):
    """ממיר בלוק של key=value למילון לפי רכב"""
    specs = {}
    current_car = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):  # שם רכב
            current_car = line[:-1]
            specs[current_car] = {"base_price_new": 100000, "fuel_efficiency": 14, "turbo": False}
        elif "=" in line and current_car:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "base_price_new":
                try:
                    specs[current_car][k] = int(v.replace(",", "").replace("₪", ""))
                except:
                    pass
            elif k == "fuel_efficiency":
                try:
                    specs[current_car][k] = float(v)
                except:
                    pass
            elif k == "turbo":
                specs[current_car][k] = v.lower() in ["true", "yes", "1"]
    return specs

def ask_perplexity_for_specs(car_list, max_retries=3):
    if not car_list:
        return {}

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    car_queries = "\n".join([f"- {car['model']} {car['year']}" for car in car_list])
    query = f"""
עבור כל אחד מהרכבים הבאים, החזר נתונים בפורמט key=value בלבד:

{car_queries}

פורמט חובה:
<Model Year>:
base_price_new=<int>
fuel_efficiency=<float>
turbo=<true/false>

אם אין נתונים – השתמש בברירת מחדל:
base_price_new=100000
fuel_efficiency=14
turbo=false

אסור להחזיר טקסט חופשי, רק את הפורמט.
"""

    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": query}]}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            raw = resp.json()
            st.text_area(f"==== RAW PERPLEXITY RESPONSE (attempt {attempt+1}) ====",
                         json.dumps(raw, ensure_ascii=False, indent=2), height=200)

            text = raw["choices"][0]["message"]["content"]
            specs = parse_kv_format_block(text)
            return specs
        except Exception as e:
            st.warning(f"⚠️ Perplexity ניסיון {attempt+1} נכשל: {e}")
    return {}

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
        calc_price = car.get("calculated_price")
        if calc_price is None:
            continue
        if not (answers["budget_min"] * 0.87 <= calc_price <= answers["budget_max"] * 1.13):
            continue
        if "turbo" in answers and car.get("turbo") is not None:
            if answers["turbo_pref"] == "כן" and not car.get("turbo"):
                continue
            if answers["turbo_pref"] == "לא" and car.get("turbo"):
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
    turbo_pref = st.selectbox("מנוע טורבו?", ["לא משנה", "כן", "לא"])
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
        "turbo_pref": turbo_pref,
        "reliability_pref": reliability_pref,
    }

    st.info("📤 שולח בקשה ל־GPT...")
    gpt_models = ask_gpt_for_models(answers)

    final_cars = []
    dict_cars, fallback_cars = [], []

    for car in gpt_models:
        brand_raw = car["model"].split()[0]
        brand = BRAND_TRANSLATION.get(brand_raw, brand_raw)
        if brand in BRAND_DICT:
            car["brand"] = brand
            dict_cars.append(car)
        else:
            fallback_cars.append(car)

    # ✅ מותגים מהמילון
    for car in dict_cars:
        params = BRAND_DICT[car["brand"]]
        calc_price = calculate_price(
            100000,
            car["year"],
            params["category"],
            params["reliability"],
            params["demand"],
            14
        )
        car["calculated_price"] = calc_price
        final_cars.append(car)

    # ✅ מותגים לא במילון → Perplexity (בבת אחת)
    if fallback_cars:
        specs_fb = ask_perplexity_for_specs(fallback_cars)
        for car in fallback_cars:
            extra = specs_fb.get(f"{car['model']} {car['year']}", {})
            calc_price = calculate_price(
                extra.get("base_price_new", 100000),
                car["year"],
                extra.get("category", "משפחתיות"),
                extra.get("reliability", "בינונית"),
                extra.get("demand", "בינוני"),
                extra.get("fuel_efficiency", 14)
            )
            car["calculated_price"] = calc_price
            car["turbo"] = extra.get("turbo", False)
            car["citations"] = extra.get("citations", [])
            final_cars.append(car)

    # סינון
    filtered = filter_results(final_cars, answers)

    if filtered:
        st.success("✅ נמצאו רכבים מתאימים:")
        df = pd.DataFrame(filtered)
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("⬇️ הורד כ־CSV", data=csv, file_name="car_results.csv", mime="text/csv")

        for car in filtered:
            if car.get("citations"):
                st.markdown(f"**מקורות עבור {car['model']} {car['year']}:**")
                for url in car["citations"]:
                    st.markdown(f"- [קישור]({url})")

    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")