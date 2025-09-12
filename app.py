import streamlit as st
import pandas as pd
import json
import os
import datetime
import google.generativeai as genai

# -----------------------------
# הגדרות API (משתמש בסיקרטס של Streamlit)
# -----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# מילון 50 מותגים נפוצים בישראל
# -----------------------------
brand_dict = {
    "Toyota": {"reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True},
    "Hyundai": {"reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Mazda": {"reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True},
    "Kia": {"reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Suzuki": {"reliability": "גבוהה", "demand": "גבוה", "luxury": False, "popular": True},
    "Nissan": {"reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Honda": {"reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": True},
    "Mitsubishi": {"reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Chevrolet": {"reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Ford": {"reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Skoda": {"reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Seat": {"reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Volkswagen": {"reliability": "בינונית", "demand": "גבוה", "luxury": False, "popular": True},
    "Peugeot": {"reliability": "בינונית", "demand": "בינוני", "luxury": False, "popular": True},
    "Renault": {"reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Opel": {"reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Fiat": {"reliability": "נמוכה", "demand": "נמוך", "luxury": False, "popular": False},
    "Subaru": {"reliability": "גבוהה", "demand": "בינוני", "luxury": False, "popular": True},
    "BMW": {"reliability": "בינונית", "demand": "גבוה", "luxury": True, "popular": True},
    "Mercedes": {"reliability": "גבוהה", "demand": "גבוה", "luxury": True, "popular": True},
    # ... להמשיך עד 50 חברות ...
}

# -----------------------------
# פונקציות עזר
# -----------------------------
def build_prompt_with_dict(cars):
    return f"""
אתה מקבל רשימת רכבים בפורמט JSON.
עבור כל רכב החזר JSON עם שני ערכים בלבד:
- base_price_new (מחיר השקה חדש בישראל בשקלים)
- fuel_efficiency (צריכת דלק בק״מ לליטר)

קלט:
{json.dumps(cars, ensure_ascii=False, indent=2)}

פלט:
"""

def build_prompt_full(cars):
    return f"""
אתה מקבל רשימת רכבים בפורמט JSON.
עבור כל רכב החזר JSON מלא עם הערכים:
- base_price_new
- category
- brand_country
- reliability
- demand
- luxury
- popular
- fuel_efficiency

קלט:
{json.dumps(cars, ensure_ascii=False, indent=2)}

פלט:
"""

def depreciation_formula(base_price, year, category, reliability, demand, fuel_efficiency):
    current_year = datetime.datetime.now().year
    age = current_year - year
    price = base_price

    # ירידת ערך בסיסית: 5% לשנה
    price *= (0.95 ** age)

    # התאמות לפי קטגוריה
    if category in ["מנהלים", "יוקרה", "SUV"]:
        price *= 0.85
    elif category in ["משפחתי"]:
        price *= 0.90
    else:
        price *= 0.92

    # אמינות
    if reliability == "גבוהה":
        price *= 1.05
    elif reliability == "נמוכה":
        price *= 0.90

    # ביקוש
    if demand == "גבוה":
        price *= 1.05
    elif demand == "נמוך":
        price *= 0.90

    # חיסכון בדלק
    if fuel_efficiency >= 18:
        price *= 1.05
    elif fuel_efficiency <= 12:
        price *= 0.95

    return round(price, -2)

def log_debug(step, data):
    log_path = "car_advisor_logs.csv"
    df = pd.DataFrame([[datetime.datetime.now(), step, json.dumps(data, ensure_ascii=False)]],
                      columns=["timestamp", "step", "data"])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

# -----------------------------
# טעינת מאגר
# -----------------------------
@st.cache_data
def load_dataset():
    path = "car_models_israel_clean.csv"
    return pd.read_csv(path)

dataset = load_dataset()

# -----------------------------
# ממשק המשתמש – שאלון
# -----------------------------
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

budget_min = st.number_input("תקציב מינימלי (₪)", min_value=1000, step=1000)
budget_max = st.number_input("תקציב מקסימלי (₪)", min_value=5000, step=1000)
fuel_type = st.selectbox("מנוע מועדף:", ["בנזין", "דיזל", "היברידי", "חשמלי"])
engine_min = st.number_input("נפח מנוע מינימלי (סמ״ק)", min_value=800, step=100)
engine_max = st.number_input("נפח מנוע מקסימלי (סמ״ק)", min_value=1000, step=100)
year_min = st.number_input("שנת ייצור מינימלית:", min_value=1990, max_value=2025, step=1)
year_max = st.number_input("שנת ייצור מקסימלית:", min_value=1990, max_value=2025, step=1)
car_type = st.selectbox("סוג רכב:", ["סדאן", "האצ׳בק", "סטיישן", "SUV", "מיניוואן", "קופה"])
gearbox = st.selectbox("גיר:", ["לא משנה", "אוטומט", "ידני"])
importance = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "חיסכון בדלק", "שמירת ערך עתידית"])

if st.button("מצא רכבים מתאימים"):
    try:
        # שלב 1: הצעת דגמים ראשוניים לפי השאלון
        candidate_cars = dataset[
            (dataset["price"] >= budget_min * 0.87) &
            (dataset["price"] <= budget_max * 1.13) &
            (dataset["engine_cc"] >= engine_min) &
            (dataset["engine_cc"] <= engine_max) &
            (dataset["year"] >= year_min) &
            (dataset["year"] <= year_max) &
            (dataset["fuel"] == fuel_type) &
            (dataset["type"] == car_type)
        ].head(5)

        if candidate_cars.empty:
            st.warning("❌ לא נמצאו רכבים מתאימים במאגר.")
        else:
            cars_for_prompt = {f"{row['brand']} {row['model']} {row['year']}": {} 
                               for _, row in candidate_cars.iterrows()}

            # שלב 2: בדיקה אם במילון או לא
            brands_in_dict = all(row["brand"] in brand_dict for _, row in candidate_cars.iterrows())
            if brands_in_dict:
                prompt = build_prompt_with_dict(cars_for_prompt)
            else:
                prompt = build_prompt_full(cars_for_prompt)

            log_debug("Gemini Prompt", prompt)

            response = model.generate_content(prompt)
            gemini_data = json.loads(response.text)

            log_debug("Gemini Response", gemini_data)

            results = []
            for car, data in gemini_data.items():
                brand = car.split()[0]
                year = int(car.split()[-1])

                # אם במילון – מחברים את הערכים ממנו
                if brand in brand_dict:
                    full_data = {**data, **brand_dict[brand]}
                else:
                    full_data = data

                price = depreciation_formula(
                    base_price=full_data["base_price_new"],
                    year=year,
                    category=full_data["category"],
                    reliability=full_data["reliability"],
                    demand=full_data["demand"],
                    fuel_efficiency=full_data["fuel_efficiency"]
                )
                results.append({"car": car, "final_price": price})

            # שלב 3: סינון לפי טווח
            filtered = [r for r in results if budget_min <= r["final_price"] <= budget_max]

            if not filtered:
                st.warning("❌ אחרי סינון מחיר לא נמצאו התאמות.")
            else:
                st.success("✅ נמצאו רכבים מתאימים:")
                st.table(filtered)

    except Exception as e:
        st.error(f"שגיאה: {e}")
        log_debug("Error", str(e))
