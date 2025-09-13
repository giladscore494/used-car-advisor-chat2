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
# 📖 BRAND DICTIONARY – חלקי
# =======================
BRAND_DICT = {
    "Toyota": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Hyundai": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Mazda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Kia": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Honda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Ford": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Volkswagen": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True, "category": "משפחתי"},
    "Nissan": {"brand_country": "יפן", "reliability": "בינונית", "demand": "נמוך", "luxury": False, "popular": True, "category": "משפחתי"},
    "Peugeot": {"brand_country": "צרפת", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Skoda": {"brand_country": "צ'כיה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True, "category": "משפחתי"},
    "Opel": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Renault": {"brand_country": "צרפת", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Subaru": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Seat": {"brand_country": "ספרד", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": False, "category": "משפחתי"},
    "Citroen": {"brand_country": "צרפת", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False, "category": "משפחתי"},
    "Mitsubishi": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": True, "category": "משפחתי"},
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
        "gearbox": "<string>"
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
            st.success(f"✅ GPT החזיר JSON תקין ({len(models)} רכבים)")
            return models
        except Exception as e:
            st.warning(f"⚠️ GPT ניסיון {attempt+1} נכשל: {e}")
    st.error("❌ GPT נכשל בכל הניסיונות")
    return []

# =======================
# 🤖 GEMINI – השלמת נתונים עם grounding
# =======================
def ask_gemini_for_specs(car_list, use_dict=True, max_retries=5):
    if not car_list:
        return {}

    if use_dict:
        prompt_template = """
        מצא את *מחיר ההשקה בישראל* לשנתון ואת צריכת הדלק הממוצעת.
        החזר JSON במבנה:
        {{
          "<model> <year>": {{
            "base_price_new": <int>,
            "fuel_efficiency": <int>
          }}
        }}
        עבור:
        {cars}
        """
    else:
        prompt_template = """
        מצא את *מחיר ההשקה בישראל* לשנתון, ופרטים נוספים.
        החזר JSON במבנה:
        {{
          "<model> <year>": {{
            "base_price_new": <int>,
            "category": "<string>",
            "brand_country": "<string>",
            "reliability": "<string>",
            "demand": "<string>",
            "luxury": <bool>,
            "popular": <bool>,
            "fuel_efficiency": <int>
          }}
        }}
        עבור:
        {cars}
        """

    model = genai.GenerativeModel("gemini-1.5-flash")

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format(cars=json.dumps(car_list, ensure_ascii=False))
            resp = model.generate_content(prompt)
            raw = resp.text.strip()
            st.text_area(f"==== RAW GEMINI RESPONSE (attempt {attempt+1}) ====", raw, height=200)

            if raw.startswith("```"):
                raw = raw.strip("```json").strip("```").strip()
            specs = json.loads(raw)
            st.success(f"✅ Gemini החזיר JSON תקין בניסיון {attempt+1}")
            return specs
        except Exception as e:
            st.warning(f"⚠️ Gemini ניסיון {attempt+1} נכשל: {e}")

    st.error("❌ Gemini נכשל 5 פעמים. משתמש בערכי ברירת מחדל.")
    specs = {}
    for car in car_list:
        specs[f"{car['model']} {car['year']}"] = {
            "base_price_new": 100000,
            "fuel_efficiency": 14,
            "category": "משפחתיות",
            "brand_country": "לא ידוע",
            "reliability": "בינונית",
            "demand": "בינוני",
            "luxury": False,
            "popular": True
        }
    return specs

# =======================
# 📉 נוסחת ירידת ערך (עם debug)
# =======================
def calculate_price(base_price_new, year, category, reliability, demand, fuel_efficiency):
    age = datetime.now().year - int(year)
    st.write(f"📉 חישוב ירידת ערך: base={base_price_new}, year={year}, age={age}, cat={category}, rel={reliability}, demand={demand}, eff={fuel_efficiency}")
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
# 🔎 סינון (עם debug)
# =======================
def filter_results(cars, answers):
    st.write(f"🔎 לפני סינון: {len(cars)} רכבים")
    filtered = []
    for car in cars:
        calc_price = car.get("calculated_price")
        if calc_price is None:
            continue
        if not (answers["budget_min"] * 0.87 <= calc_price <= answers["budget_max"] * 1.13):
            continue
        filtered.append(car)
    st.write(f"🔎 אחרי סינון: {len(filtered)} רכבים")
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

    st.info("📤 שולח בקשה ל־GPT...")
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
        specs_dict = ask_gemini_for_specs(dict_cars, use_dict=True)
        for car in dict_cars:
            brand = car["model"].split()[0]
            params = BRAND_DICT[brand]
            extra = specs_dict.get(f"{car['model']} {car['year']}", {})
            car["calculated_price"] = calculate_price(
                extra.get("base_price_new", 100000),
                car["year"],
                params["category"],
                params["reliability"],
                params["demand"],
                extra.get("fuel_efficiency", 14)
            )
            final_cars.append(car)

    if fallback_cars:
        specs_fb = ask_gemini_for_specs(fallback_cars, use_dict=False)
        for car in fallback_cars:
            extra = specs_fb.get(f"{car['model']} {car['year']}", {})
            car["calculated_price"] = calculate_price(
                extra.get("base_price_new", 100000),
                car["year"],
                extra.get("category", "משפחתיות"),
                extra.get("reliability", "בינונית"),
                extra.get("demand", "בינוני"),
                extra.get("fuel_efficiency", 14)
            )
            final_cars.append(car)

    filtered = filter_results(final_cars, answers)

    if filtered:
        st.success("✅ נמצאו רכבים מתאימים:")
        st.dataframe(pd.DataFrame(filtered))
    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")
