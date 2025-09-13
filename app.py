import os
import re
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import requests
from rapidfuzz import fuzz

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
    "Toyota": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "popular": True, "category": "משפחתי"},
    "Hyundai": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "popular": True, "category": "משפחתי"},
    "Mazda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "popular": True, "category": "משפחתי"},
    "Kia": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "popular": True, "category": "משפחתי"},
    "Honda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "popular": False, "category": "משפחתי"},
    "Ford": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "popular": False, "category": "משפחתי"},
    "Volkswagen": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "popular": True, "category": "משפחתי"},
    "Audi": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "popular": True, "category": "יוקרה"},
    "BMW": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "popular": True, "category": "יוקרה"},
    "Mercedes": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "popular": True, "category": "יוקרה"},
    "Suzuki": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "popular": True, "category": "סופר מיני"},
}

BRAND_TRANSLATION = {
    "יונדאי": "Hyundai", "מאזדה": "Mazda", "טויוטה": "Toyota", "קיה": "Kia",
    "הונדה": "Honda", "פורד": "Ford", "פולקסווגן": "Volkswagen", "אודי": "Audi",
    "ב.מ.וו": "BMW", "מרצדס": "Mercedes", "סוזוקי": "Suzuki",
}

# =======================
# 🧠 GPT – בחירת דגמים
# =======================
def ask_gpt_for_models(user_answers, max_retries=3):
    prompt = f"""
    בהתבסס על השאלון הבא, הצע עד 20 דגמים רלוונטיים בישראל.
    כל דגם חייב להתאים לדרישות (כולל טורבו אם סונן).
    החזר טבלה (לא JSON!) עם עמודות:
    model | year | engine_cc | fuel | gearbox | turbo

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

            # ננסה להמיר לטבלה → DF → records
            try:
                df = pd.read_csv(pd.compat.StringIO(raw), sep="|").dropna(axis=1, how="all")
                return df.to_dict(orient="records")
            except Exception:
                pass
        except Exception as e:
            st.warning(f"⚠️ GPT ניסיון {attempt+1} נכשל: {e}")
    return []

# =======================
# 🌐 PERPLEXITY – Specs
# =======================
def ask_perplexity_for_specs(car_list):
    if not car_list:
        return pd.DataFrame()

    query = "החזר טבלה עם העמודות: Model | Year | Base Price New | Fuel Efficiency | Turbo.\n"
    query += "המידע חייב לכלול את כל הדגמים: " + ", ".join([f"{c['model']} {c['year']}" for c in car_list])

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": query}]}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = resp.json()
        st.text_area("==== RAW PERPLEXITY RESPONSE ====", json.dumps(raw, ensure_ascii=False, indent=2), height=250)
        text = raw["choices"][0]["message"]["content"]

        # ננסה לקרוא כטבלה
        df_specs = pd.read_csv(pd.compat.StringIO(text), sep="|").dropna(axis=1, how="all")
        return df_specs
    except Exception as e:
        st.error(f"❌ Perplexity נכשל: {e}")
        return pd.DataFrame()

# =======================
# 📉 נוסחת ירידת ערך חדשה
# =======================
def calculate_price(base_price_new, year, category, brand_country, reliability, demand, popular, fuel_efficiency):
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
# 🔎 סינון גמיש + Debug
# =======================
def filter_results(cars, answers, df_specs):
    filtered = []
    debug_log = []

    for car in cars:
        reason = []

        # מאגר
        matches = [m for m in car_db["model"].values if fuzz.partial_ratio(str(car["model"]), str(m)) > 75]
        if not matches:
            reason.append("❌ לא נמצא במאגר")
            debug_log.append((car["model"], reason))
            continue

        # נתוני Perplexity
        row = df_specs[df_specs["Model"].str.contains(car["model"].split()[0], case=False, na=False)]
        if row.empty:
            reason.append("❌ אין נתונים מ־Perplexity")
            debug_log.append((car["model"], reason))
            continue

        try:
            base_price = int(str(row["Base Price New"].values[0]).replace("₪", "").replace(",", "").strip())
            fuel_eff = float(str(row["Fuel Efficiency"].values[0]).split()[0])
            turbo_flag = str(row["Turbo"].values[0]).lower() in ["true", "yes", "כן"]
        except Exception:
            reason.append("❌ נתונים לא ניתנים להמרה")
            debug_log.append((car["model"], reason))
            continue

        brand = car["model"].split()[0]
        params = BRAND_DICT.get(brand, {"category": "משפחתיות", "brand_country": "יפן", "reliability": "בינונית",
                                        "demand": "בינוני", "popular": False})

        _, calc_price, _ = calculate_price(base_price, int(car["year"]),
                                           params["category"], params["brand_country"],
                                           params["reliability"], params["demand"],
                                           params["popular"], fuel_eff)

        # סינון מחיר
        if not (answers["budget_min"] <= calc_price <= answers["budget_max"]):
            reason.append("❌ נפל בסינון מחיר")
            debug_log.append((car["model"], reason))
            continue

        # סינון טורבו
        if answers["turbo"] == "כן" and not turbo_flag:
            reason.append("❌ אין טורבו")
            debug_log.append((car["model"], reason))
            continue
        if answers["turbo"] == "לא" and turbo_flag:
            reason.append("❌ יש טורבו")
            debug_log.append((car["model"], reason))
            continue

        car["calculated_price"] = calc_price
        filtered.append(car)

    return filtered, debug_log

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
        "fuel": fuel, "gearbox": gearbox,
        "body_type": body_type, "turbo": turbo,
        "reliability_pref": reliability_pref, "extra_notes": extra_notes
    }

    st.info("📤 שולח בקשה ל־GPT...")
    gpt_models = ask_gpt_for_models(answers)

    st.info("🌐 שולח בקשה ל־Perplexity...")
    df_specs = ask_perplexity_for_specs(gpt_models)

    st.info("🔎 סינון תוצאות...")
    filtered, debug_log = filter_results(gpt_models, answers, df_specs)

    if filtered:
        st.success("✅ נמצאו רכבים מתאימים:")
        st.dataframe(pd.DataFrame(filtered))
    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")

    st.subheader("⚙️ Debug Log")
    for car, reasons in debug_log:
        st.markdown(f"**{car}** → {', '.join(reasons)}")