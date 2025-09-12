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
# 📖 BRAND DICTIONARY – מותגים נפוצים בישראל
# =======================
BRAND_DICT = {
    "Toyota": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Hyundai": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Mazda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Kia": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Honda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Ford": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Volkswagen": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "משפחתי"},
    "Skoda": {"brand_country": "צ׳כיה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Suzuki": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "סופר מיני"},
}

# =======================
# 🧠 GPT – בחירת דגמים
# =======================
def ask_gpt_for_models(user_answers):
    prompt = f"""
    בהתבסס על השאלון הבא, הצע עד 20 דגמים רלוונטיים בישראל.
    החזר JSON בלבד, בפורמט:
    [
      {{
        "model": "<string>",
        "year": <int>,
        "engine_cc": <int>,
        "fuel": "<string>",
        "gearbox": "<string>"
      }}
    ]

    שאלון:
    {json.dumps(user_answers, ensure_ascii=False)}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    raw = response.choices[0].message.content
    st.write("==== RAW GPT RESPONSE ====")
    st.code(raw)

    try:
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        st.error(f"❌ לא נמצא JSON תקין בתשובת GPT: {e}")
        return []

# =======================
# 🤖 GEMINI – השלמת נתונים עם לולאת תיקון
# =======================
def ask_gemini_for_specs(car_list, use_dict=True, max_retries=5):
    if use_dict:
        prompt = f"""
        החזר JSON תקין בלבד, בלי טקסט נוסף.
        עבור כל דגם החזר:
        - base_price_new
        - fuel_efficiency
        {json.dumps(car_list, ensure_ascii=False)}
        """
    else:
        prompt = f"""
        החזר JSON תקין בלבד, בלי טקסט נוסף.
        עבור כל דגם החזר:
        - base_price_new
        - category
        - brand_country
        - reliability
        - demand
        - luxury
        - popular
        - fuel_efficiency
        {json.dumps(car_list, ensure_ascii=False)}
        """

    model = genai.GenerativeModel("gemini-1.5-flash")

    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt)
            raw = resp.text.strip()
            st.write(f"==== RAW GEMINI RESPONSE (ניסיון {attempt+1}) ====")
            st.code(raw)

            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            st.warning(f"⚠️ JSONDecodeError בניסיון {attempt+1}: {e}")
            continue

    st.error("❌ לא התקבל JSON תקין מג׳מיני אחרי 5 ניסיונות")
    return {}

# =======================
# 📉 נוסחת ירידת ערך
# =======================
def calculate_price(base_price_new, year, category, reliability, demand, fuel_efficiency):
    age = datetime.now().year - int(year)
    price = base_price_new
    price *= (1 - 0.07) ** age
    if category in ["מנהלים", "יוקרה"]: price *= 0.85
    elif category in ["מיני", "סופר מיני"]: price *= 0.95
    if reliability == "גבוהה": price *= 1.05
    elif reliability == "נמוכה": price *= 0.9
    if demand == "גבוה": price *= 1.05
    elif demand == "נמוך": price *= 0.9
    if fuel_efficiency >= 18: price *= 1.05
    elif fuel_efficiency <= 12: price *= 0.95
    if age > 10: price *= 0.85
    return round(price, -2)

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
        "budget_min": budget_min, "budget_max": budget_max,
        "engine_min": engine_min, "engine_max": engine_max,
        "year_min": year_min, "year_max": year_max,
        "fuel": fuel, "gearbox": gearbox,
        "body_type": body_type, "reliability_pref": reliability_pref
    }

    st.info("📤 שולח בקשה ל־GPT לדגמים מתאימים...")
    gpt_models = ask_gpt_for_models(answers)

    log_data = {"time": str(datetime.now()), "answers": answers, "gpt_models": gpt_models}

    if not gpt_models:
        st.error("⚠️ לא התקבלו דגמים מ־GPT")
    else:
        st.success(f"✅ התקבלו {len(gpt_models)} דגמים מ־GPT")
        st.subheader("🔍 Debug – דגמים מ־GPT")
        st.json(gpt_models)

        final_cars, dict_cars, fallback_cars = [], [], []
        for car in gpt_models:
            brand = car["model"].split()[0]
            if brand in BRAND_DICT: dict_cars.append(car)
            else: fallback_cars.append(car)

        st.info(f"✅ במילון: {len(dict_cars)} | ⚠️ פולבאק: {len(fallback_cars)}")
        st.subheader("📊 Debug – חלוקה מילון / פולבאק")
        st.json({"dict_cars": dict_cars, "fallback_cars": fallback_cars})

        if dict_cars:
            specs_dict = ask_gemini_for_specs(dict_cars, use_dict=True)
            log_data["gemini_dict"] = specs_dict
            st.subheader("📊 Debug – תשובות Gemini (מילון)")
            st.json(specs_dict)
            for car in dict_cars:
                brand = car["model"].split()[0]
                params = BRAND_DICT[brand]
                extra = specs_dict.get(f"{car['model']} {car['year']}", {})
                if not extra: continue
                calc_price = calculate_price(extra["base_price_new"], car["year"], params["category"], params["reliability"], params["demand"], extra["fuel_efficiency"])
                car["calculated_price"] = calc_price
                final_cars.append(car)

        if fallback_cars:
            specs_fb = ask_gemini_for_specs(fallback_cars, use_dict=False)
            log_data["gemini_fallback"] = specs_fb
            st.subheader("📊 Debug – תשובות Gemini (פולבאק)")
            st.json(specs_fb)
            for car in fallback_cars:
                extra = specs_fb.get(f"{car['model']} {car['year']}", {})
                if not extra: continue
                calc_price = calculate_price(extra["base_price_new"], car["year"], extra["category"], extra["reliability"], extra["demand"], extra["fuel_efficiency"])
                car["calculated_price"] = calc_price
                final_cars.append(car)

        log_data["final_cars"] = final_cars
        st.subheader("📊 Debug – מחירים אחרי נוסחה")
        st.json(final_cars)

        if final_cars:
            st.success("✅ נמצאו רכבים מתאימים:")
            st.dataframe(pd.DataFrame(final_cars))
        else:
            st.error("⚠️ לא נמצאו רכבים מתאימים.")

    # ✍️ כתיבת לוג לקובץ
    with open("car_advisor_logs.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")