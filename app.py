import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

from openai import OpenAI
import google.generativeai as genai

# --- טעינת סודות ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --- טעינת מאגר מקומי ---
DATA_PATH = "car_models_israel_clean.csv"
car_df = pd.read_csv(DATA_PATH)

# --- מילון מותגים (50 חברות נפוצות בישראל) ---
BRAND_DICT = {
    "Toyota": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True},
    "Hyundai": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Mazda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True},
    "Kia": {"brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Suzuki": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": True},
    "Honda": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": True},
    "Chevrolet": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Ford": {"brand_country": "ארה״ב", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Skoda": {"brand_country": "צ׳כיה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Seat": {"brand_country": "ספרד", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Volkswagen": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Peugeot": {"brand_country": "צרפת", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Renault": {"brand_country": "צרפת", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Opel": {"brand_country": "גרמניה", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Fiat": {"brand_country": "איטליה", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Subaru": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": True},
    "BMW": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True},
    "Mercedes": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True},
    "Audi": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True},
    "Volvo": {"brand_country": "שוודיה", "reliability": "גבוהה", "demand": "בינוני", "luxury": True, "popular": True},
    "Jaguar": {"brand_country": "בריטניה", "reliability": "נמוכה", "demand": "נמוך", "luxury": True, "popular": False},
    "Land Rover": {"brand_country": "בריטניה", "reliability": "נמוכה", "demand": "נמוך", "luxury": True, "popular": False},
    "Jeep": {"brand_country": "ארה״ב", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Chrysler": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Dodge": {"brand_country": "ארה״ב", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Alfa Romeo": {"brand_country": "איטליה", "reliability": "נמוכה", "demand": "נמוך", "luxury": True, "popular": False},
    "Mitsubishi": {"brand_country": "יפן", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Citroen": {"brand_country": "צרפת", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Mini": {"brand_country": "בריטניה", "reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True},
    "Porsche": {"brand_country": "גרמניה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True},
    "Tesla": {"brand_country": "ארה״ב", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True},
    "Saab": {"brand_country": "שוודיה", "reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Cadillac": {"brand_country": "ארה״ב", "reliability": "בינונית", "demand": "נמוך", "luxury": True, "popular": False},
    "Infiniti": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "בינוני", "luxury": True, "popular": False},
    "Lexus": {"brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True},
    "Acura": {"brand_country": "יפן", "reliability": "בינונית", "demand": "נמוך", "luxury": True, "popular": False},
    "Genesis": {"brand_country": "קוריאה", "reliability": "גבוהה", "demand": "בינוני", "luxury": True, "popular": False},
    "BYD": {"brand_country": "סין", "reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Chery": {"brand_country": "סין", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Geely": {"brand_country": "סין", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "MG": {"brand_country": "סין", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Great Wall": {"brand_country": "סין", "reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": False},
    "DS": {"brand_country": "צרפת", "reliability": "בינונית", "demand": "בינוני", "luxury": True, "popular": False},
    "Smart": {"brand_country": "גרמניה", "reliability": "בינונית", "demand": "נמוך", "luxury": False, "popular": False},
    "Maserati": {"brand_country": "איטליה", "reliability": "נמוכה", "demand": "נמוך", "luxury": True, "popular": False},
    "Ferrari": {"brand_country": "איטליה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": False},
    "Lamborghini": {"brand_country": "איטליה", "reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": False}
}

# --- נוסחת ירידת ערך ---
def calculate_price(base_price_new, year, category, reliability, demand, fuel_efficiency):
    age = datetime.now().year - year
    price = base_price_new
    price *= (0.95 ** age)
    if category in ["מנהלים", "יוקרה", "SUV"]:
        price *= 0.85
    elif category in ["משפחתי"]:
        price *= 0.90
    else:
        price *= 0.92
    if reliability == "גבוהה":
        price *= 1.05
    elif reliability == "נמוכה":
        price *= 0.90
    if demand == "גבוה":
        price *= 1.05
    elif demand == "נמוך":
        price *= 0.90
    if fuel_efficiency >= 18:
        price *= 1.05
    elif fuel_efficiency <= 12:
        price *= 0.95
    return max(round(price, -2), 2000)

# --- GPT: בחירת דגמים ---
def ask_gpt_models(user_answers):
    prompt = f"""
אתה עוזר מומחה לרכבים בישראל.
על סמך הנתונים הבאים, החזר עד 15 דגמים אמיתיים שנמכרו בישראל בלבד.
אסור להחזיר דגמים שלא שווקו בישראל.
אסור טקסט חופשי, הערות או סימני ```.

חוקי היגיון:
- רכב ישן (2005–2010, עממי) כיום 5,000–40,000 ₪.
- משפחתי חדש (2018+) 60,000–200,000 ₪.
- רכבי יוקרה חדשים (BMW/Mercedes וכו׳) מעל 150,000 ₪.
- מחירים סבירים ביחס לשוק הישראלי.

שדות חובה:
- "model"
- "year"
- "engine_cc"
- "fuel"
- "gearbox"

קלט שאלון:
{json.dumps(user_answers, ensure_ascii=False, indent=2)}

פלט JSON:
[
  {{"model": "Toyota Corolla", "year": 2017, "engine_cc": 1600, "fuel": "בנזין", "gearbox": "אוטומט"}},
  {{"model": "Hyundai i30", "year": 2016, "engine_cc": 1600, "fuel": "בנזין", "gearbox": "אוטומט"}},
  {{"model": "Mazda 3", "year": 2015, "engine_cc": 2000, "fuel": "בנזין", "gearbox": "ידני"}}
]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "אתה עוזר מומחה לרכבים."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return []

# --- Gemini: בקשה אחת לכל הדגמים ---
def ask_gemini_specs_batch(cars):
    prompt = f"""
אתה מקבל רשימת רכבים בפורמט JSON.
החזר JSON בלבד (אסור טקסט חופשי או סימני ```).

לכל רכב החזר:
- base_price_new (מחיר ההשקה בישראל בשקלים, מספר בלבד)
- category (מיני/סופר מיני/משפחתי/מנהלים/יוקרה/SUV/קופה/מיניוואן)
- brand_country (מדינת מותג)
- reliability (גבוהה/בינונית/נמוכה)
- demand (גבוה/בינוני/נמוך)
- luxury (true/false)
- popular (true/false)
- fuel_efficiency (צריכת דלק בק״מ לליטר, מספר בלבד)

חוקי היגיון למחירים:
- מיני/סופר מיני: 70,000–120,000 ₪ בהשקה.
- משפחתי: 110,000–150,000 ₪ בהשקה.
- מנהלים/SUV: 150,000–300,000 ₪ בהשקה.
- יוקרה: מעל 250,000 ₪ בהשקה.

קלט:
{json.dumps(cars, ensure_ascii=False, indent=2)}

פלט JSON:
{{
  "Toyota Corolla 2017": {{"base_price_new": 132000, "category": "משפחתי", "brand_country": "יפן", "reliability": "גבוהה", "demand": "גבוה", "luxury": false, "popular": true, "fuel_efficiency": 15}},
  "Hyundai i30 2016": {{"base_price_new": 118000, "category": "משפחתי", "brand_country": "קוריאה", "reliability": "בינונית", "demand": "גבוה", "luxury": false, "popular": true, "fuel_efficiency": 14}}
}}
"""
    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
    try:
        return json.loads(raw)
    except Exception:
        return {}

# --- ממשק משתמש ---
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

with st.form("user_form"):
    budget_min = st.number_input("תקציב מינימלי (₪)", 0, 300000, 20000)
    budget_max = st.number_input("תקציב מקסימלי (₪)", 0, 300000, 50000)
    fuel_pref = st.selectbox("מנוע מועדף:", ["בנזין", "דיזל", "היברידי", "חשמלי"])
    engine_min = st.number_input("נפח מנוע מינימלי (סמ״ק)", 600, 5000, 1200)
    engine_max = st.number_input("נפח מנוע מקסימלי (סמ״ק)", 600, 5000, 1800)
    year_min = st.number_input("שנת ייצור מינימלית", 1990, 2025, 2010)
    year_max = st.number_input("שנת ייצור מקסימלית", 1990, 2025, 2020)
    body_type = st.selectbox("סוג רכב:", ["סדאן", "האצ׳בק", "SUV", "מיני"])
    gearbox = st.selectbox("גיר:", ["לא משנה", "אוטומט", "ידני"])
    use_case = st.selectbox("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    reliability_pref = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "חסכון", "שמירת ערך"])
    submitted = st.form_submit_button("מצא רכבים")

if submitted:
    user_answers = {
        "budget_min": budget_min,
        "budget_max": budget_max,
        "fuel_pref": fuel_pref,
        "engine_min": engine_min,
        "engine_max": engine_max,
        "year_min": year_min,
        "year_max": year_max,
        "body_type": body_type,
        "gearbox": gearbox,
        "use_case": use_case,
        "priority": reliability_pref,
    }

    st.info("🔎 מחפש דגמים מתאימים...")
    gpt_models = ask_gpt_models(user_answers)

    if not gpt_models:
        st.error("❌ לא נמצאו דגמים מתאימים בשלב הראשוני.")
    else:
        cars_for_prompt = {f"{row['model']} {row['year']}": {} for row in gpt_models}
        gemini_data = ask_gemini_specs_batch(cars_for_prompt)

        results = []
        for car, specs in gemini_data.items():
            try:
                brand = car.split()[0]
                year = int(car.split()[-1])
                model = " ".join(car.split()[:-1])

                base_price_new = specs.get("base_price_new", 100000)
                category = specs.get("category", "משפחתי")
                fuel_eff = specs.get("fuel_efficiency", 14)

                brand_data = BRAND_DICT.get(brand, {})
                reliability = brand_data.get("reliability", specs.get("reliability", "בינונית"))
                demand = brand_data.get("demand", specs.get("demand", "בינוני"))
                brand_country = brand_data.get("brand_country", specs.get("brand_country", "לא ידוע"))
                luxury = brand_data.get("luxury", specs.get("luxury", False))
                popular = brand_data.get("popular", specs.get("popular", False))

                calc_price = calculate_price(base_price_new, year, category, reliability, demand, fuel_eff)

                if not (budget_min <= calc_price <= budget_max):
                    continue

                results.append({
                    "דגם": model,
                    "שנה": year,
                    "מחיר נוכחי": f"{calc_price:,} ₪",
                    "מחיר חדש בהשקה": f"{base_price_new:,} ₪",
                    "סגמנט": category,
                    "אמינות": reliability,
                    "ביקוש": demand,
                    "יוקרה": "כן" if luxury else "לא",
                    "פופולריות": "כן" if popular else "לא",
                    "צריכת דלק (ק״מ/ל׳)": fuel_eff,
                    "מדינת מותג": brand_country,
                })
            except Exception:
                continue

        if results:
            st.success("✅ נמצאו רכבים מתאימים:")
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("⚠️ לא נמצאו רכבים מתאימים לאחר חישוב מחיר.")

        # לוג
        log_entry = {"timestamp": datetime.now().isoformat(), "answers": user_answers, "results": results}
        with open("car_advisor_logs.csv", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
