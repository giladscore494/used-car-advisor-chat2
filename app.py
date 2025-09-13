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
# 🧮 נוסחת ירידת ערך חדשה
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
# 🧠 GPT – בחירת דגמים
# =======================
def ask_gpt_for_models(user_answers, max_retries=3):
    prompt = f"""
    על סמך התשובות לשאלון, החזר עד 20 רכבים מתאימים בישראל.
    החזר JSON בלבד:
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

    חובה להתאים: תקציב, שנים, נפח מנוע, דלק, גיר, סוג רכב, טורבו.
    הערות חופשיות: {user_answers.get('extra_notes', '')}
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.strip("```json").strip("```").strip()

            models = json.loads(raw)
            st.write(f"⚙️ Debug: GPT attempt {attempt+1} החזיר {len(models)} דגמים")
            return models
        except Exception as e:
            st.write(f"⚙️ Debug: GPT attempt {attempt+1} נכשל → {e}")
    return []

# =======================
# 🌐 PERPLEXITY BULK
# =======================
def ask_perplexity_bulk(car_list, max_retries=2):
    if not car_list:
        return pd.DataFrame()

    query = "החזר CSV עם עמודות: model,year,base_price_new,fuel_efficiency,turbo.\n"
    for car in car_list:
        query += f"- {car['model']} {car['year']}\n"

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": query}]}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            raw = resp.json()
            text = raw["choices"][0]["message"]["content"]
            csv_match = re.search(r"(model,year,base_price_new,fuel_efficiency,turbo[\s\S]+)", text)
            if csv_match:
                from io import StringIO
                df = pd.read_csv(StringIO(csv_match.group(1)))
                st.write(f"⚙️ Debug: Perplexity attempt {attempt+1} הצליח → {len(df)} שורות")
                return df
            else:
                st.write(f"⚙️ Debug: Perplexity attempt {attempt+1} לא מצא CSV")
        except Exception as e:
            st.write(f"⚙️ Debug: Perplexity attempt {attempt+1} נכשל → {e}")

    return pd.DataFrame()

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
        "budget_min": budget_min, "budget_max": budget_max,
        "engine_min": engine_min, "engine_max": engine_max,
        "year_min": year_min, "year_max": year_max,
        "fuel": fuel, "gearbox": gearbox, "body_type": body_type,
        "turbo": turbo, "reliability_pref": reliability_pref,
        "extra_notes": extra_notes
    }
    st.write("⚙️ Debug: תשובות משתמש", answers)

    gpt_models = ask_gpt_for_models(answers)
    specs_df = ask_perplexity_bulk(gpt_models)

    final_cars = []
    for _, row in specs_df.iterrows():
        brand = row["model"].split()[0]
        brand = BRAND_TRANSLATION.get(brand, brand)
        params = BRAND_DICT.get(brand, {"brand_country": "לא ידוע", "reliability": "בינונית",
                                        "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"})

        price_low, price_est, price_high = calculate_price(
            row["base_price_new"], int(row["year"]),
            params["category"], params["brand_country"],
            params["reliability"], params["demand"],
            params["popular"], row["fuel_efficiency"]
        )

        final_cars.append({
            "model": row["model"], "year": int(row["year"]),
            "brand": brand, "base_price_new": row["base_price_new"],
            "fuel_efficiency": row["fuel_efficiency"], "turbo": row["turbo"],
            "price_low": price_low, "calculated_price": price_est, "price_high": price_high
        })
        st.write(f"⚙️ Debug: חישוב מחיר → {row['model']} {row['year']} → {price_est}₪")

    if final_cars:
        df = pd.DataFrame(final_cars)
        st.success(f"✅ נמצאו {len(df)} רכבים מתאימים")
        st.dataframe(df)
        st.download_button("⬇️ הורד כ־CSV", data=df.to_csv(index=False), file_name="car_results.csv", mime="text/csv")
    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים")