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
# 🌐 PERPLEXITY – השלמת נתוני רכב
# =======================
def parse_price_and_fuel(text):
    base_price, fuel_eff, turbo = 100000, 14, False
    price_match = re.search(r"(\d{2,3}[.,]?\d{0,3}) ?ש״?ח", text)
    fuel_match = re.search(r"(\d{1,2}[.,]?\d?) ?ליטר ל-?100", text)
    turbo_match = re.search(r"טורבו|turbo|TSI|TFSI|TURBO", text, re.IGNORECASE)

    if price_match:
        base_price = int(price_match.group(1).replace(",", "").replace(".", ""))
    if fuel_match:
        fuel_eff = float(fuel_match.group(1))
    if turbo_match:
        turbo = True

    return base_price, fuel_eff, turbo

def ask_perplexity_for_specs(car_list, max_retries=5):
    if not car_list:
        return {}

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    specs = {}
    queries = []
    for car in car_list:
        query = f"{car['model']} {car['year']} מחיר השקה בישראל, צריכת דלק ממוצעת (ליטרים ל-100 ק״מ), האם יש טורבו. החזר JSON עם base_price_new, fuel_efficiency, turbo."
        queries.append({"role": "user", "content": query})

    payload = {"model": "sonar-pro", "messages": queries}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = resp.json()
        st.text_area("==== RAW PERPLEXITY RESPONSE ====", json.dumps(raw, ensure_ascii=False, indent=2), height=200)

        text = raw["choices"][0]["message"]["content"]

        try:
            parsed_all = json.loads(text)
        except:
            parsed_all = {}

        for car in car_list:
            key = f"{car['model']} {car['year']}"
            parsed = parsed_all.get(key, {})
            base_price = parsed.get("base_price_new", 100000)
            fuel_eff = parsed.get("fuel_efficiency", 14)
            turbo = parsed.get("turbo", False)

            specs[key] = {
                "base_price_new": base_price,
                "fuel_efficiency": fuel_eff,
                "turbo": turbo,
                "citations": raw.get("citations", [])
            }

    except Exception as e:
        st.warning(f"⚠️ Perplexity נכשל: {e}")
        for car in car_list:
            specs[f"{car['model']} {car['year']}"] = {
                "base_price_new": 100000,
                "fuel_efficiency": 14,
                "turbo": False,
                "citations": []
            }

    return specs

# =======================
# 📉 נוסחת ירידת ערך
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

    return price_low, price_est, price_high

# =======================
# 🔎 סינון
# =======================
def filter_results(cars, answers):
    filtered = []
    for car in cars:
        calc_price = car.get("calculated_price")
        if calc_price is None:
            continue

        # ✅ סינון לפי תקציב
        if not (answers["budget_min"] * 0.87 <= calc_price[1] <= answers["budget_max"] * 1.13):
            continue

        # ✅ סינון לפי טורבו
        turbo_pref = answers.get("turbo", "לא משנה")
        car_turbo = str(car.get("turbo", "")).lower() in ["true", "1", "yes", "כן"]

        if turbo_pref == "כן" and not car_turbo:
            continue
        if turbo_pref == "לא" and car_turbo:
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
    turbo_pref = st.selectbox("מנוע טורבו", ["לא משנה", "כן", "לא"])
    reliability_pref = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "חיסכון בדלק", "שמירת ערך"])
    extra_notes = st.text_area("הערות חופשיות (אופציונלי)")
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
        "turbo": turbo_pref,
        "reliability_pref": reliability_pref,
        "extra_notes": extra_notes,
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

    for car in dict_cars:
        params = BRAND_DICT[car["brand"]]
        car["calculated_price"] = calculate_price(
            100000,
            car["year"],
            params["category"],
            params["brand_country"],
            params["reliability"],
            params["demand"],
            params["popular"],
            14
        )
        final_cars.append(car)

    if fallback_cars:
        specs_fb = ask_perplexity_for_specs(fallback_cars)
        for car in fallback_cars:
            extra = specs_fb.get(f"{car['model']} {car['year']}", {})
            car["calculated_price"] = calculate_price(
                extra.get("base_price_new", 100000),
                car["year"],
                extra.get("category", "משפחתיות"),
                extra.get("brand_country", "יפן"),
                extra.get("reliability", "בינונית"),
                extra.get("demand", "בינוני"),
                extra.get("popular", False),
                extra.get("fuel_efficiency", 14)
            )
            car["turbo"] = extra.get("turbo", False)
            car["citations"] = extra.get("citations", [])
            final_cars.append(car)

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