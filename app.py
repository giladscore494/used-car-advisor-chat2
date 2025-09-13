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
# 🧮 נוסחת ירידת ערך חדשה
# =======================
def calculate_price(base_price_new, year, category, brand_country,
                    reliability, demand, popular, fuel_efficiency):
    current_year = datetime.now().year
    age = current_year - year

    # ירידת ערך בסיסית לפי גיל
    if age <= 5:
        depreciation_rate = 0.10
    elif age <= 10:
        depreciation_rate = 0.15
    else:
        depreciation_rate = 0.22

    # התאמות לפי קטגוריה/מותג
    if category in ["יוקרה", "מנהלים"] or brand_country in ["גרמניה", "ארה״ב"]:
        depreciation_rate += 0.03
    elif brand_country in ["יפן", "קוריאה"]:
        depreciation_rate -= 0.02

    # ביקוש
    if demand == "גבוה":
        depreciation_rate -= 0.02
    elif demand == "נמוך":
        depreciation_rate += 0.02

    # אמינות
    if reliability == "גבוהה":
        depreciation_rate -= 0.02
    elif reliability == "נמוכה":
        depreciation_rate += 0.03

    # חישוב מחיר משוער
    price_est = base_price_new * ((1 - depreciation_rate) ** age)

    # מינימום מחיר רצפה
    price_est = max(price_est, 5000)

    # טווח מחיר
    price_low = int(price_est * 0.9)
    price_high = int(price_est * 1.1)

    return price_low, price_est, price_high

# =======================
# 📋 פונקציית שליפה קשיחה (GPT/Perplexity)
# =======================
def fetch_with_retries(query_func, user_answers, max_retries=5):
    prompt = f"""
    על סמך הקריטריונים:
    {json.dumps(user_answers, ensure_ascii=False)}

    החזר אך ורק טבלה בפורמט Markdown (לא JSON, לא טקסט חופשי) עם העמודות:
    | Model | Year | Base Price New | Fuel Efficiency | Turbo |

    דרישות:
    - התחל את הפלט ישר מהטבלה (הסימן הראשון חייב להיות '|').
    - כל שורה מייצגת רכב.
    - בעמודת Turbo יש רק true או false.
    - בעמודת Year רק מספר ארבע ספרות.
    - אם אין מידע → החזר טבלה ריקה עם הכותרות בלבד.
    """

    for attempt in range(max_retries):
        raw = query_func(prompt)
        raw = raw.strip()
        if raw.startswith("|") and "Model" in raw and "Year" in raw:
            return raw
    return "| Model | Year | Base Price New | Fuel Efficiency | Turbo |\n|-------|------|----------------|-----------------|-------|\n"

# =======================
# 🌐 GPT API
# =======================
def gpt_api_call(prompt):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"| Model | Year | Base Price New | Fuel Efficiency | Turbo |\n|-------|------|----------------|-----------------|-------|\n"

# =======================
# 🌐 PERPLEXITY API
# =======================
def perplexity_api_call(prompt):
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(url, headers=headers, json=payload, timeout=40)
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"| Model | Year | Base Price New | Fuel Efficiency | Turbo |\n|-------|------|----------------|-----------------|-------|\n"

# =======================
# 🛠️ Parse Table
# =======================
def parse_table(markdown_text):
    try:
        lines = [l for l in markdown_text.splitlines() if l.strip().startswith("|")]
        headers = [h.strip() for h in lines[0].split("|")[1:-1]]
        rows = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            rows.append(cells)
        return pd.DataFrame(rows, columns=headers)
    except Exception:
        return pd.DataFrame(columns=["Model", "Year", "Base Price New", "Fuel Efficiency", "Turbo"])

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
    extra_notes = st.text_area("הערות חופשיות (אופציונלי)", "")
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
    raw_gpt = fetch_with_retries(gpt_api_call, answers)
    df_gpt = parse_table(raw_gpt)
    st.text_area("==== RAW GPT RESPONSE ====", raw_gpt, height=200)

    st.info("📤 שולח בקשה ל־Perplexity...")
    raw_px = fetch_with_retries(perplexity_api_call, answers)
    df_px = parse_table(raw_px)
    st.text_area("==== RAW PERPLEXITY RESPONSE ====", raw_px, height=200)

    final_df = pd.concat([df_gpt, df_px], ignore_index=True).drop_duplicates()

    if not final_df.empty:
        st.success("✅ נמצאו רכבים מתאימים:")
        st.dataframe(final_df)
        csv = final_df.to_csv(index=False)
        st.download_button("⬇️ הורד כ־CSV", data=csv, file_name="car_results.csv", mime="text/csv")
    else:
        st.error("⚠️ לא נמצאו רכבים מתאימים.")