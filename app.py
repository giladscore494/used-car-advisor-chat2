import os
import re
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import requests
from io import StringIO

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
        "turbo": <bool>
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
# 🌐 PERPLEXITY – בקשה אחת עם טבלה
# =======================
def ask_perplexity_for_specs(car_list, max_retries=3):
    if not car_list:
        return pd.DataFrame()

    # רשימת דגמים
    car_lines = "\n".join([f"- {c['model']} {c['year']}" for c in car_list])

    query = f"""
    עבור הרשימה הבאה של רכבים, מצא את הנתונים הבאים באינטרנט:
    1. מחיר ההשקה בישראל (base price new, ₪).
    2. צריכת דלק ממוצעת (liters per 100 km).
    3. האם יש טורבו (true/false).

    החזר אך ורק כטבלה טקסטואלית בפורמט Markdown עם כותרות:
    Model | Year | Base Price New | Fuel Efficiency | Turbo

    רשימת רכבים:
    {car_lines}
    """

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": query}]}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            raw = resp.json()
            st.text_area(f"==== RAW PERPLEXITY RESPONSE (attempt {attempt+1}) ====",
                         json.dumps(raw, ensure_ascii=False, indent=2), height=250)

            text = raw["choices"][0]["message"]["content"]

            # ניקוי ``` אם קיים
            cleaned = text.strip().replace("```", "")
            if cleaned.lower().startswith("markdown"):
                cleaned = "\n".join(cleaned.split("\n")[1:])

            # המרה ל-DataFrame
            df = pd.read_csv(StringIO(cleaned), sep="|").apply(lambda x: x.str.strip() if x.dtype=="object" else x)
            return df
        except Exception as e:
            st.warning(f"⚠️ Perplexity ניסיון {attempt+1} נכשל: {e}")
    return pd.DataFrame()

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
    turbo = st.selectbox("מנוע טורבו", ["לא משנה", "כן", "לא"])
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
        "turbo": turbo,
        "reliability_pref": reliability_pref,
        "extra_notes": extra_notes
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
        _, calc_price, _ = calculate_price(
            100000,
            car["year"],
            params["category"],
            params["brand_country"],
            params["reliability"],
            params["demand"],
            params["popular"],
            14
        )
        car["calculated_price"] = calc_price
        final_cars.append(car)

    if fallback_cars:
        df_specs = ask_perplexity_for_specs(fallback_cars)
        if not df_specs.empty:
            for car in fallback_cars:
                row = df_specs[df_specs["Model"].str.contains(car["model"].split()[0], case=False, na=False)]
                if not row.empty:
                    try:
                        base_price_new = int(str(row["Base Price New"].values[0]).replace(",", "").replace("₪", "").strip())
                    except:
                        base_price_new = 100000
                    try:
                        fuel_eff = float(str(row["Fuel Efficiency"].values[0]).replace(",", ".").strip())
                    except:
                        fuel_eff = 14
                    turbo_val = str(row["Turbo"].values[0]).lower() in ["true", "yes", "כן"]

                    calc_low, calc_est, calc_high = calculate_price(
                        base_price_new,
                        int(car["year"]),
                        "משפחתיות",
                        "יפן",
                        "בינונית",
                        "בינוני",
                        False,
                        fuel_eff
                    )
                    car["calculated_price"] = calc_est
                    car["turbo_detected"] = turbo_val
                    final_cars.append(car)

    filtered = filter_results(final_cars, answers)

    if filtered:
        st.success("✅ נמצאו רכבים מתאימים:")
        df = pd.DataFrame(filtered)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("⬇️ הורד כ־CSV", data=csv, file_name="car_results.csv", mime="text/csv")
    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")