
import os
import re
import json
import requests
import datetime
import streamlit as st
import pandas as pd
import unidecode
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
# קריאה בטוחה ל-JSON
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
# שלב 1 – Gemini מציע רשימת דגמים
# =============================
def fetch_candidate_models(answers):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                החזר רשימה של לפחות 10 דגמים מתאימים
                בפורמט JSON פשוט:
                ["Model1", "Model2", "Model3", ...]
                אל תוסיף טקסט נוסף מעבר ל-JSON.
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 2 – סינון מול משרד התחבורה
# =============================
def normalize_name(name: str) -> str:
    return unidecode.unidecode(str(name)).lower().replace(" ", "").replace("-", "")

def filter_models_by_registry(candidate_models, answers, df_cars):
    valid_models = []
    df_cars["model_norm"] = df_cars["model"].astype(str).apply(normalize_name)

    for model_name in candidate_models:
        norm = normalize_name(model_name)
        exists = df_cars[df_cars["model_norm"].str.contains(norm, na=False)]
        if exists.empty:
            continue

        # גיר
        if answers.get("gearbox") == "אוטומט":
            if exists["automatic"].max() != 1:
                continue

        # דלק
        if answers.get("engine") and answers["engine"] != "לא משנה":
            fuels = exists["fuel"].unique().tolist()
            if not any(answers["engine"] in f for f in fuels):
                continue

        # שנת ייצור
        if answers.get("year_range"):
            year_range = answers["year_range"].replace("+", "").split("–")
            year_min, year_max = [int(y) for y in year_range]
            years = exists["year"].astype(int)
            if not any((years >= year_min) & (years <= year_max)):
                continue

        valid_models.append(model_name)

    return valid_models

# =============================
# שלב 3 – Gemini מחזיר נתונים יבשים
# =============================
def fetch_models_data_with_gemini(valid_models, answers):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                אלו הדגמים שנבחרו אחרי סינון מול משרד התחבורה:
                {valid_models}

                החזר נתונים יבשים עבור כל דגם בפורמט JSON:
                {{
                  "Model Name": {{
                     "price_range": "טווח מחירון ביד שנייה (₪)",
                     "availability": "זמינות בישראל",
                     "insurance_total": "עלות ביטוח חובה + צד ג' (₪, טווח עם דיסקליימר)",
                     "license_fee": "אגרת רישוי/טסט שנתית (₪)",
                     "maintenance": "תחזוקה שנתית ממוצעת (₪)",
                     "common_issues": "תקלות נפוצות",
                     "fuel_consumption": "צריכת דלק אמיתית (ק״מ לליטר)",
                     "depreciation": "ירידת ערך ממוצעת (%)",
                     "safety": "דירוג בטיחות (כוכבים)",
                     "parts_availability": "זמינות חלפים בישראל"
                  }}
                }}
                אל תוסיף טקסט נוסף מעבר ל-JSON.
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
    - הסבר יתרונות וחסרונות של כל אחד
    - התייחס במיוחד לעלות ביטוח, תחזוקה, ירידת ערך וצריכת דלק
    - הצג את הסיבות למה הם הכי מתאימים לתקציב ולצרכים של המשתמש
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        temperature=0.4,
    )
    return response.choices[0].message.content

# =============================
# שמירת לוג
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

# טען מאגר משרד התחבורה
df_cars = pd.read_csv("car_models_israel.csv")

# Session state
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "summary_text" not in st.session_state:
    st.session_state["summary_text"] = None

with st.form("car_form"):
    answers = {}
    answers["budget_min"] = int(st.text_input("תקציב מינימלי (₪)", "20000"))
    answers["budget_max"] = int(st.text_input("תקציב מקסימלי (₪)", "50000"))
    answers["engine"] = st.radio("מנוע מועדף:", ["לא משנה", "בנזין", "דיזל", "היברידי", "חשמלי"])
    answers["engine_size"] = st.selectbox("נפח מנוע (סמ״ק):", ["לא משנה", "1200", "1600", "2000", "3000+"])
    answers["year_range"] = st.selectbox("שנות ייצור:", ["2010–2015", "2016–2020", "2021+"])
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'בק", "SUV", "טנדר", "משפחתי"])
    answers["gearbox"] = st.radio("גיר:", ["לא משנה", "אוטומט", "ידני"])
    answers["usage"] = st.radio("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    answers["size"] = st.selectbox("גודל רכב:", ["קטן", "משפחתי", "SUV", "טנדר"])
    answers["driver_age"] = st.selectbox("גיל הנהג הראשי:", ["18–20", "21–24", "25–30", "31–40", "40+"])
    answers["license_years"] = st.selectbox("ותק רישיון נהיגה:", ["פחות משנה", "1–3", "4–7", "8+"])
    answers["insurance_history"] = st.selectbox("עבר ביטוחי/תעבורתי:", ["ללא תביעות/תאונות/דוחות", "תביעה אחת", "ריבוי תביעות"])
    answers["annual_km"] = st.selectbox("נסועה שנתית (ק״מ):", ["פחות מ-10,000", "10,000–20,000", "20,000–30,000", "30,000+"])
    answers["passengers"] = st.selectbox("מספר נוסעים עיקרי:", ["לרוב לבד", "2–3", "משפחה מלאה"])
    answers["maintenance_budget"] = st.selectbox("יכולת השקעה בתחזוקה שנתית:", ["פחות מ-3,000 ₪", "3,000–5,000 ₪", "מעל 5,000 ₪"])
    answers["reliability_vs_comfort"] = st.radio("מה חשוב יותר?", ["אמינות וחיסכון", "נוחות/ביצועים גם במחיר תחזוקה"])
    answers["eco"] = st.radio("שיקולי איכות סביבה:", ["לא משנה", "חשוב מאוד"])
    answers["resale_value"] = st.radio("שמירת ערך עתידית:", ["חשוב", "פחות חשוב"])
    answers["extra"] = st.text_area("משהו נוסף?")

    submitted = st.form_submit_button("שלח וקבל המלצה")

if submitted:
    with st.spinner("🌐 Gemini מחפש דגמים מתאימים..."):
        candidate_models = fetch_candidate_models(answers)
    st.markdown("### 📝 דגמים ש-Gemini הציע")
    st.write(candidate_models)

    with st.spinner("🧹 סינון מול משרד התחבורה..."):
        valid_models = filter_models_by_registry(candidate_models, answers, df_cars)
    st.markdown("### ✅ דגמים אחרי סינון משרד התחבורה")
    st.write(valid_models)

    with st.spinner("📊 Gemini מחזיר נתונים יבשים..."):
        models_data = fetch_models_data_with_gemini(valid_models, answers)

    try:
        df = pd.DataFrame(models_data).T
        df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
        st.session_state["results_df"] = df
    except Exception as e:
        st.warning("⚠️ בעיה בנתוני JSON")
        st.write(models_data)

    with st.spinner("⚡ GPT מסכם ומדרג..."):
        summary = final_recommendation_with_gpt(answers, models_data)
        st.session_state["summary_text"] = summary

    save_log(answers, models_data, summary)

# הצגת תוצאות מה-Session State
if st.session_state["results_df"] is not None:
    st.subheader("📊 השוואת נתונים בין הדגמים")
    st.dataframe(st.session_state["results_df"], use_container_width=True)
    csv = st.session_state["results_df"].to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד כ-CSV", csv, "car_advisor.csv", "text/csv")

if st.session_state["summary_text"] is not None:
    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary_text"])
