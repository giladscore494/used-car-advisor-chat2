import os
import re
import json
import requests
import datetime
import streamlit as st
import pandas as pd
from openai import OpenAI

# =============================
# מפתחות API
# =============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    st.error("❌ לא נמצאו מפתחות API. ודא שהגדרת אותם ב-secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# קריאה בטוחה ל-Gemini
# =============================
def safe_gemini_call(payload, model="gemini-2.0-flash"):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    try:
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=120)
        data = r.json()
        if "candidates" not in data:
            return f"שגיאת Gemini: {data}"
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"שגיאה: {e}"

# =============================
# פיענוח JSON
# =============================
def parse_gemini_json(answer):
    cleaned = answer.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {}

# =============================
# שלב 1 – סינון ראשוני מול מאגר משרד התחבורה
# =============================
def filter_with_mot(answers, mot_file="car_models_israel_clean.csv"):
    if not os.path.exists(mot_file):
        st.error(f"❌ קובץ המאגר '{mot_file}' לא נמצא בתיקייה. ודא שהעלית אותו.")
        return []

    df = pd.read_csv(mot_file)

    # המרות בטוחות
    for col in ["year", "engine_cc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    year_min = int(answers["year_min"])
    year_max = int(answers["year_max"])
    cc_min = int(answers["engine_cc_min"])
    cc_max = int(answers["engine_cc_max"])

    mask_year = df["year"].between(year_min, year_max, inclusive="both")
    mask_cc = df["engine_cc"].between(cc_min, cc_max, inclusive="both")

    # סינון לפי סוג דלק
    mask_fuel = df["fuel_normalized"] == answers["engine"]

    # סינון לפי גיר
    if answers["gearbox"] == "אוטומט":
        mask_gear = df["automatic"].astype(int) == 1
    elif answers["gearbox"] == "ידני":
        mask_gear = df["automatic"].astype(int) == 0
    else:
        mask_gear = pd.Series([True] * len(df), index=df.index)

    df_filtered = df[mask_year & mask_cc & mask_fuel & mask_gear].copy()

    return df_filtered.to_dict(orient="records")

# =============================
# שלב 2 – Gemini מוסיף נתונים יבשים
# =============================
def fetch_models_from_gemini(answers, verified_models):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                הנה רשימת רכבים שעברו סינון ראשוני:
                {verified_models}

                עבור כל רכב החזר אך ורק JSON עם השדות הבאים, נתונים לישראל בלבד:
                {{
                  "Model (year, engine, fuel)": {{
                     "price_range": "טווח מחירון ממוצע ביד שנייה (₪)",
                     "insurance_total": "עלות ביטוח חובה + צד ג' (₪)",
                     "maintenance": "תחזוקה שנתית ממוצעת (₪)",
                     "common_issues": "תקלות נפוצות",
                     "depreciation": "ירידת ערך ממוצעת (%)",
                     "safety": "דירוג בטיחות (כוכבים)",
                     "parts_availability": "זמינות חלפים בישראל",
                     "turbo": 0/1
                  }}
                }}

                ❌ אל תוסיף דגמים חדשים.
                ❌ אל תסנן לפי שאלון.
                ✅ החזר אך ורק נתונים יבשים מהמקורות.
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 3 – GPT מסנן ומדרג
# =============================
def final_recommendation_with_gpt(answers, enriched_data):
    text = f"""
    תשובות המשתמש:
    {answers}

    נתוני הדגמים (מגימניי):
    {enriched_data}

    צור סיכום בעברית:
    - בחר עד 5 דגמים בלבד
    - סנן לפי תקציב מחירון, תחזוקה, ירידת ערך, שימוש עיקרי, אמינות/נוחות, שמירת ערך, ביטוח, איכות סביבה
    - פרט יתרונות וחסרונות
    - צור טבלת 10 פרמטרים סופית
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        temperature=0.4,
    )
    return response.choices[0].message.content

# =============================
# שלב 4 – Cache (שמירה חודשית)
# =============================
def save_cache(enriched_data, filename="car_data_cache.csv"):
    df_new = pd.DataFrame.from_dict(enriched_data, orient="index")
    df_new["update_date"] = datetime.date.today().isoformat()

    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        final = df_new

    final.to_csv(filename, index=False, encoding="utf-8-sig")

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Car-Advisor", page_icon="🚗")
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

with st.form("car_form"):
    answers = {}
    answers["budget_min"] = int(st.text_input("תקציב מינימלי (₪)", "5000"))
    answers["budget_max"] = int(st.text_input("תקציב מקסימלי (₪)", "20000"))
    answers["engine"] = st.radio("מנוע מועדף:", ["בנזין", "דיזל", "היברידי-בנזין", "היברידי-דיזל", "חשמל"])
    answers["engine_cc_min"] = int(st.text_input("נפח מנוע מינימלי (סמ״ק):", "1200"))
    answers["engine_cc_max"] = int(st.text_input("נפח מנוע מקסימלי (סמ״ק):", "2000"))
    answers["year_min"] = st.text_input("שנת ייצור מינימלית:", "2000")
    answers["year_max"] = st.text_input("שנת ייצור מקסימלית:", "2020")
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'בק", "SUV", "מיני", "סופר מיני", "סטיישן", "טנדר", "משפחתי"])
    answers["gearbox"] = st.radio("גיר:", ["לא משנה", "אוטומט", "ידני"])
    answers["turbo"] = st.radio("מנוע טורבו:", ["לא משנה", "כן", "לא"])
    answers["usage"] = st.radio("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    answers["driver_age"] = st.selectbox("גיל הנהג הראשי:", ["עד 21", "21–24", "25–34", "35+"])
    answers["license_years"] = st.selectbox("ותק רישיון נהיגה:", ["פחות משנה", "1–3 שנים", "3–5 שנים", "מעל 5 שנים"])
    answers["insurance_history"] = st.selectbox("עבר ביטוחי/תעבורתי:", ["ללא", "תאונה אחת", "מספר תביעות"])
    answers["annual_km"] = st.selectbox("נסועה שנתית (ק״מ):", ["עד 10,000", "10,000–20,000", "20,000–30,000", "מעל 30,000"])
    answers["passengers"] = st.selectbox("מספר נוסעים עיקרי:", ["לרוב לבד", "2 אנשים", "3–5 נוסעים", "מעל 5"])
    answers["maintenance_budget"] = st.selectbox("יכולת תחזוקה:", ["מתחת 3,000 ₪", "3,000–5,000 ₪", "מעל 5,000 ₪"])
    answers["reliability_vs_comfort"] = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "איזון אמינות ונוחות", "נוחות/ביצועים"])
    answers["eco_pref"] = st.selectbox("שיקולי איכות סביבה:", ["חשוב רכב ירוק/חסכוני", "לא משנה"])
    answers["resale_value"] = st.selectbox("שמירת ערך עתידית:", ["חשוב לשמור על ערך", "פחות חשוב"])
    answers["extra"] = st.text_area("משהו נוסף שתרצה לציין?")
    submitted = st.form_submit_button("שלח וקבל המלצה")

if submitted:
    with st.spinner("📊 סינון קשיח מול המאגר..."):
        verified_models = filter_with_mot(answers)

    with st.spinner("🌐 Gemini מוסיף נתונים יבשים..."):
        enriched_data = fetch_models_from_gemini(answers, verified_models)

    with st.spinner("⚡ GPT מסנן ומדרג..."):
        summary = final_recommendation_with_gpt(answers, enriched_data)

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(summary)

    # שמירה ב-Cache
    save_cache(enriched_data)
