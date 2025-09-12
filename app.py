import os
import json
import streamlit as st
import google.generativeai as genai
from openai import OpenAI

# =======================
# 🔑 מפתחות API
# =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# =======================
# 📖 מילון מותגים עם פרמטרים
# =======================
brand_dict = {
    "טויוטה": {"brand_country": "יפן", "reliability": 9, "demand": 9, "luxury": "עממי", "popular": True},
    "מאזדה": {"brand_country": "יפן", "reliability": 8, "demand": 8, "luxury": "עממי", "popular": True},
    "יונדאי": {"brand_country": "דרום קוריאה", "reliability": 8, "demand": 8, "luxury": "עממי", "popular": True},
    "קיה": {"brand_country": "דרום קוריאה", "reliability": 8, "demand": 8, "luxury": "עממי", "popular": True},
    "פולקסווגן": {"brand_country": "גרמניה", "reliability": 7, "demand": 8, "luxury": "עממי", "popular": True},
    "סקודה": {"brand_country": "צ'כיה", "reliability": 7, "demand": 7, "luxury": "עממי", "popular": True},
    "סיאט": {"brand_country": "ספרד", "reliability": 7, "demand": 7, "luxury": "עממי", "popular": True},
    "פיאט": {"brand_country": "איטליה", "reliability": 6, "demand": 6, "luxury": "עממי", "popular": False},
    "שברולט": {"brand_country": "ארה״ב", "reliability": 6, "demand": 7, "luxury": "עממי", "popular": True},
    "אופל": {"brand_country": "גרמניה", "reliability": 6, "demand": 6, "luxury": "עממי", "popular": True},
    "רנו": {"brand_country": "צרפת", "reliability": 6, "demand": 6, "luxury": "עממי", "popular": True},
    "פיג'ו": {"brand_country": "צרפת", "reliability": 6, "demand": 6, "luxury": "עממי", "popular": True},
    "סוזוקי": {"brand_country": "יפן", "reliability": 7, "demand": 6, "luxury": "עממי", "popular": True},
    "הונדה": {"brand_country": "יפן", "reliability": 8, "demand": 7, "luxury": "עממי", "popular": True},
    "פורד": {"brand_country": "ארה״ב", "reliability": 6, "demand": 7, "luxury": "עממי", "popular": True},
    "BMW": {"brand_country": "גרמניה", "reliability": 7, "demand": 9, "luxury": "יוקרתי", "popular": True},
    "מרצדס": {"brand_country": "גרמניה", "reliability": 7, "demand": 9, "luxury": "יוקרתי", "popular": True},
    "אודי": {"brand_country": "גרמניה", "reliability": 7, "demand": 8, "luxury": "יוקרתי", "popular": True},
    "וולוו": {"brand_country": "שבדיה", "reliability": 7, "demand": 7, "luxury": "יוקרתי", "popular": True},
    # אפשר להרחיב עד 50 מותגים
}

# =======================
# 🧠 GPT – הצעת דגמים (עם retry וניקוי JSON)
# =======================
def ask_gpt_for_models(answers, max_retries=5):
    prompt = f"""
    על בסיס הדרישות של המשתמש:
    תקציב: {answers['budget_min']}–{answers['budget_max']} ₪
    נפח מנוע: {answers['engine_min']}–{answers['engine_max']} סמ״ק
    שנת ייצור: {answers['year_min']}–{answers['year_max']}
    מנוע מועדף: {answers['fuel']}
    גיר: {answers['gearbox']}
    סוג רכב: {answers['car_type']}
    עדיפות: {answers['priority']}

    החזר אך ורק JSON חוקי עם מערך רכבים, כל רכב:
    {{
      "model": "<string>",
      "year": <int>,
      "engine_cc": <int>,
      "fuel": "<string>",
      "gearbox": "<string>"
    }}
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            raw_text = response.choices[0].message.content.strip()

            st.write(f"==== RAW GPT RESPONSE (attempt {attempt+1}) ====")
            st.code(raw_text, language="json")

            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").replace("json", "", 1).strip()

            models = json.loads(raw_text)
            return models
        except Exception as e:
            st.warning(f"❌ GPT attempt {attempt+1} failed: {e}")

    st.error("❌ GPT לא הצליח להחזיר JSON חוקי.")
    return []

# =======================
# 🧠 GEMINI – השלמת נתונים (עם retry וניקוי JSON)
# =======================
def fetch_specs_from_gemini(model_name, year, max_retries=5):
    prompt = f"""
    מצא נתוני רכב עבור הדגם הבא:
    דגם: {model_name}, שנה: {year}

    החזר אך ורק JSON חוקי:
    {{
      "price_range": [<int>, <int>],
      "hp": <int>,
      "torque": <int>,
      "reliability_score": <float>,
      "safety_score": <float>
    }}
    """

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            raw_text = response.text.strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").replace("json", "", 1).strip()

            if not raw_text.startswith("{"):
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1:
                    raw_text = raw_text[start:end+1]

            specs = json.loads(raw_text)

            st.write(f"==== RAW GEMINI RESPONSE (attempt {attempt+1}) ====")
            st.code(raw_text, language="json")
            return specs
        except Exception as e:
            st.warning(f"❌ Gemini attempt {attempt+1} failed: {e}")

    st.error(f"❌ Gemini לא הצליח להחזיר JSON חוקי עבור {model_name} {year}.")
    return None

# =======================
# 🚗 Streamlit App
# =======================
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

with st.form("car_form"):
    budget_min = st.number_input("תקציב מינימלי (₪)", 5000, 200000, 10000, step=1000)
    budget_max = st.number_input("תקציב מקסימלי (₪)", 5000, 200000, 15000, step=1000)
    engine_min = st.number_input("נפח מנוע מינימלי (סמ״ק)", 800, 5000, 1200, step=100)
    engine_max = st.number_input("נפח מנוע מקסימלי (סמ״ק)", 800, 5000, 1800, step=100)
    year_min = st.number_input("שנת ייצור מינימלית", 1995, 2025, 2010)
    year_max = st.number_input("שנת ייצור מקסימלית", 1995, 2025, 2016)
    fuel = st.selectbox("מנוע מועדף", ["בנזין", "דיזל", "היברידי", "חשמלי"])
    gearbox = st.selectbox("גיר", ["אוטומט", "ידני"])
    car_type = st.text_input("סוג רכב (למשל: סדאן, SUV, האצ׳בק)", "סדאן")
    priority = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "חיסכון בדלק", "ביצועים", "יוקרה"])

    submitted = st.form_submit_button("מצא רכבים")

if submitted:
    answers = {
        "budget_min": budget_min,
        "budget_max": budget_max,
        "engine_min": engine_min,
        "engine_max": engine_max,
        "year_min": year_min,
        "year_max": year_max,
        "fuel": fuel,
        "gearbox": gearbox,
        "car_type": car_type,
        "priority": priority
    }

    st.write("📤 שולח בקשה ל־GPT לדגמים מתאימים...")
    models = ask_gpt_for_models(answers)

    if not models:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")
    else:
        enriched = []
        for car in models:
            brand = car["model"].split()[0]
            if brand in brand_dict:
                specs = {
                    "base_price_new": None,
                    "fuel_efficiency": None,
                    **brand_dict[brand]
                }
                st.success(f"✅ {car['model']} במילון – לוקחים פרמטרים מוכנים")
            else:
                st.warning(f"⚠️ {car['model']} לא במילון – פונה ל־Gemini")
                specs = fetch_specs_from_gemini(car["model"], car["year"])

            if specs:
                car.update(specs)
                enriched.append(car)

        if enriched:
            st.success(f"✅ נמצא מידע עבור {len(enriched)} רכבים")
            st.json(enriched)
        else:
            st.error("⚠️ לא נמצאו נתונים מתאימים אחרי העשרה.")