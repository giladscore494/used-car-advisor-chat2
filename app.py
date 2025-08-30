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
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=90)
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
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
        cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        return {"error": str(e), "raw": cleaned}

# =============================
# שלב 1 – Gemini מחזיר רשימת דגמים
# =============================
def fetch_models_list_with_gemini(answers):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                החזר רשימה של לפחות 10 דגמים שנמכרים בישראל
                אך ורק אם הם עומדים בקריטריונים:
                - מחיר {answers['budget_min']}–{answers['budget_max']} ₪
                - שנות ייצור {answers['year_range']}
                - סוג רכב {answers['car_type']}
                - מנוע {answers['engine']}
                - שימוש עיקרי {answers['usage']}

                החזר JSON בפורמט:
                ["דגם1","דגם2","דגם3",...]
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 2 – אימות מול משרד התחבורה
# =============================
def filter_models_by_mot(models_list, mot_file="car_models_israel.csv"):
    try:
        mot_df = pd.read_csv(mot_file)
        mot_models = mot_df['model_name'].dropna().unique().tolist()
        verified = [m for m in models_list if m in mot_models]
        return verified
    except Exception as e:
        return []

# =============================
# שלב 3 – Gemini מחזיר טבלה עם 10 פרמטרים
# =============================
def fetch_models_data_with_gemini(verified_models):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                בנה טבלה של הדגמים הבאים בלבד:
                {', '.join(verified_models)}

                עבור כל דגם החזר JSON עם השדות:
                {{
                  "Model Name": {{
                     "price_range": "...",
                     "availability": "...",
                     "insurance_total": "...",
                     "license_fee": "...",
                     "maintenance": "...",
                     "common_issues": "...",
                     "fuel_consumption": "...",
                     "depreciation": "...",
                     "safety": "...",
                     "parts_availability": "..."
                  }}
                }}
                החזר JSON תקני בלבד.
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 4 – GPT מסכם ומדרג
# =============================
def final_recommendation_with_gpt(answers, models_data):
    text = f"""
    תשובות המשתמש:
    {answers}

    נתוני הדגמים:
    {models_data}

    צור סיכום בעברית:
    - בחר את 5 הדגמים הטובים ביותר בלבד
    - פרט יתרונות וחסרונות
    - התייחס לעלות ביטוח, תחזוקה, ירידת ערך וצריכת דלק
    - הסבר למה הם הכי מתאימים למשתמש
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        temperature=0.4,
    )
    return response.choices[0].message.content

# =============================
# פונקציית לוג
# =============================
def save_log(answers, models_data, summary, filename="car_advisor_logs.csv"):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "answers": json.dumps(answers, ensure_ascii=False),
        "summary": summary,
        "models_data": json.dumps(models_data, ensure_ascii=False)
    }
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        new_df = pd.DataFrame([record])
        final = pd.concat([existing, new_df], ignore_index=True)
    else:
        final = pd.DataFrame([record])
    final.to_csv(filename, index=False, encoding="utf-8-sig")

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Car-Advisor", page_icon="🚗")
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

COLUMN_TRANSLATIONS = {
    "price_range": "טווח מחירון",
    "availability": "זמינות בישראל",
    "insurance_total": "ביטוח חובה+צד ג' (דיסקליימר)",
    "license_fee": "אגרת רישוי",
    "maintenance": "תחזוקה שנתית",
    "common_issues": "תקלות נפוצות",
    "fuel_consumption": "צריכת דלק",
    "depreciation": "ירידת ערך",
    "safety": "בטיחות",
    "parts_availability": "חלפים בישראל"
}

with st.form("car_form"):
    answers = {}
    answers["budget_min"] = int(st.text_input("תקציב מינימלי (₪)", "10000"))
    answers["budget_max"] = int(st.text_input("תקציב מקסימלי (₪)", "20000"))
    answers["engine"] = st.radio("מנוע מועדף:", ["בנזין", "דיזל", "היברידי", "חשמלי"])
    answers["engine_size"] = st.selectbox("נפח מנוע (סמ״ק):", ["1200", "1600", "2000", "3000+"])
    answers["year_range"] = st.selectbox("שנות ייצור:", ["2010–2015", "2016–2020", "2021+"])
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'בק", "SUV", "טנדר", "משפחתי"])
    answers["usage"] = st.radio("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    answers["driver_age"] = st.selectbox("גיל הנהג הראשי:", ["עד 21", "21–24", "25–34", "35+"])
    answers["license_years"] = st.selectbox("ותק רישיון נהיגה:", ["פחות משנה", "1–3 שנים", "3–5 שנים", "מעל 5 שנים"])
    answers["insurance_history"] = st.selectbox("עבר ביטוחי/תעבורתי:", ["ללא", "תאונה אחת", "מספר תביעות"])
    answers["maintenance_budget"] = st.selectbox("יכולת תחזוקה:", ["מתחת 3,000 ₪", "3,000–5,000 ₪", "מעל 5,000 ₪"])
    submitted = st.form_submit_button("שלח וקבל המלצה")

# =============================
# טיפול אחרי שליחה
# =============================
if submitted:
    with st.spinner("🌐 Gemini מחפש רשימת דגמים..."):
        models_list = fetch_models_list_with_gemini(answers)

    st.subheader("📝 דגמים ש-Gemini הציע")
    st.write(models_list)

    with st.spinner("✅ אימות מול משרד התחבורה..."):
        verified_models = filter_models_by_mot(models_list)
    st.subheader("דגמים אחרי סינון משרד התחבורה")
    st.write(verified_models)

    if verified_models:
        with st.spinner("🌐 Gemini בונה טבלה עם 10 פרמטרים..."):
            models_data = fetch_models_data_with_gemini(verified_models)
        try:
            df = pd.DataFrame(models_data).T
            df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
            st.session_state["df"] = df
        except Exception as e:
            st.warning("⚠️ בעיה בנתוני JSON")
            st.write(models_data)

        with st.spinner("⚡ GPT מסכם ומדרג..."):
            summary = final_recommendation_with_gpt(answers, models_data)
            st.session_state["summary"] = summary

        save_log(answers, models_data, st.session_state["summary"])
    else:
        st.warning("⚠️ לא נמצאו דגמים אחרי אימות משרד התחבורה.")

# =============================
# הצגת תוצאות
# =============================
if "df" in st.session_state:
    st.subheader("📊 השוואת נתונים בין הדגמים")
    st.dataframe(st.session_state["df"], use_container_width=True)

    csv = st.session_state["df"].to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד כ-CSV", csv, "car_advisor.csv", "text/csv")

if "summary" in st.session_state:
    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])
