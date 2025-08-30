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
# פיענוח JSON – כולל תיקון Markdown ומערכים
# =============================
def parse_gemini_json(answer):
    cleaned = answer.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
        cleaned = cleaned.replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            merged = {}
            for obj in data:
                if isinstance(obj, dict):
                    merged.update(obj)
            return merged
        return data
    except Exception as e:
        return {"error": str(e), "raw": cleaned}

# =============================
# שלב 1 – Gemini בוחר דגמים מתאימים לפי כל הקריטריונים
# =============================
def fetch_models_data_with_gemini(answers):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                בחר לפחות 10 דגמים מתאימים שנמכרים בישראל
                אך ורק אם הם עומדים בכל הקריטריונים האלו:
                - מחיר ביד שנייה בטווח {answers['budget_min']}–{answers['budget_max']} ₪
                - נפח מנוע: {answers['engine_size']} סמ״ק
                - שנות ייצור: {answers['year_range']}
                - סוג רכב: {answers['car_type']}
                - שימוש עיקרי: {answers['usage']}
                - גודל רכב: {answers['size']}
                - התאם ביטוח לפי: גיל {answers['driver_age']}, ותק {answers['license_years']}, עבר ביטוחי {answers['insurance_history']}
                - התאם תחזוקה לפי: {answers['maintenance_budget']}
                - אם המשתמש ביקש אמינות מעל הכול → עדיפות לרכבים אמינים
                - אם המשתמש ביקש שמירת ערך → עדיפות לרכבים ששומרים ערך

                עבור כל דגם החזר JSON תקני בלבד עם השדות:
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

                חובה:
                - אל תציע דגמים שלא עומדים בקריטריונים.
                - החזר מינימום 10 דגמים אם קיימים.
                - אל תוסיף טקסט מעבר ל-JSON.
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 2 – GPT מסכם ומדרג
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
# פונקציית לוג – שומר JSON מקורי עם שמות הדגמים
# =============================
def save_log(answers, models_data, summary, filename="car_advisor_logs.csv"):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "answers": json.dumps(answers, ensure_ascii=False),
        "summary": summary,
        # ✅ שומר JSON המקורי עם שמות הדגמים
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
    # שאלות בסיסיות
    answers["budget_range"] = st.selectbox("טווח תקציב:", ["5–10K", "10–20K", "20–40K", "40K+"])
    answers["budget_min"] = int(st.text_input("תקציב מינימלי (₪)", "10000"))
    answers["budget_max"] = int(st.text_input("תקציב מקסימלי (₪)", "20000"))
    answers["engine"] = st.radio("מנוע מועדף:", ["בנזין", "דיזל", "היברידי", "חשמלי"])
    answers["engine_size"] = st.selectbox("נפח מנוע (סמ״ק):", ["1200", "1600", "2000", "3000+"])
    answers["year_range"] = st.selectbox("שנות ייצור:", ["2010–2015", "2016–2020", "2021+"])
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'בק", "SUV", "טנדר", "משפחתי"])
    answers["turbo"] = st.radio("מנוע טורבו:", ["לא משנה", "כן", "לא"])
    answers["gearbox"] = st.radio("גיר:", ["לא משנה", "אוטומט", "ידני", "רובוטי"])
    answers["usage"] = st.radio("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    answers["size"] = st.selectbox("גודל רכב:", ["קטן", "משפחתי", "SUV", "טנדר"])
    
    # שאלות קריטיות נוספות
    answers["driver_age"] = st.selectbox("גיל הנהג הראשי:", ["עד 21", "21–24", "25–34", "35+"])
    answers["license_years"] = st.selectbox("ותק רישיון נהיגה:", ["פחות משנה", "1–3 שנים", "3–5 שנים", "מעל 5 שנים"])
    answers["insurance_history"] = st.selectbox("עבר ביטוחי/תעבורתי:", ["ללא תביעות/תאונות/דוחות", "תאונה אחת/דוח", "מספר תביעות/שלילה"])
    answers["annual_km"] = st.selectbox("נסועה שנתית (ק״מ):", ["עד 10,000", "10,000–20,000", "20,000–30,000", "מעל 30,000"])
    answers["passengers"] = st.selectbox("מספר נוסעים עיקרי:", ["לרוב לבד", "2 אנשים", "3–5 נוסעים", "מעל 5"])
    answers["maintenance_budget"] = st.selectbox("יכולת השקעה בתחזוקה שנתית:", ["מתחת 3,000 ₪", "3,000–5,000 ₪", "מעל 5,000 ₪"])
    answers["reliability_vs_comfort"] = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול", "איזון אמינות ונוחות", "נוחות/ביצועים גם במחיר תחזוקה"])
    answers["eco_pref"] = st.selectbox("שיקולי איכות סביבה:", ["חשוב רכב ירוק/חסכוני", "לא משנה"])
    answers["resale_value"] = st.selectbox("שמירת ערך עתידית:", ["חשוב לשמור על ערך", "פחות חשוב"])
    
    answers["extra"] = st.text_area("משהו נוסף?")

    submitted = st.form_submit_button("שלח וקבל המלצה")

# =============================
# טיפול אחרי שליחה
# =============================
if submitted:
    with st.spinner("🌐 Gemini מחפש רכבים מתאימים..."):
        models_data = fetch_models_data_with_gemini(answers)

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

    # ✅ שמירת לוג עם שמות דגמים
    try:
        save_log(answers, models_data, st.session_state["summary"])
    except Exception as e:
        st.warning(f"בעיה בשמירת הלוג: {e}")

# =============================
# הצגת תוצאות אם קיימות ב-Session
# =============================
if "df" in st.session_state:
    df = st.session_state["df"]

    def highlight_numeric(val, low_good=True):
        try:
            num = float(str(val).replace("₪", "").replace("%", "").replace(",", "").strip().split()[0])
        except:
            return ""
        if low_good:
            if num <= 3000:
                return "background-color: #d4efdf"
            elif num >= 7000:
                return "background-color: #f5b7b1"
        else:
            if num >= 16:
                return "background-color: #d4efdf"
            elif num <= 10:
                return "background-color: #f5b7b1"
        return ""

    subsets = {
        "low_good": ["ביטוח חובה+צד ג' (דיסקליימר)", "תחזוקה שנתית", "ירידת ערך"],
        "high_good": ["צריכת דלק"]
    }

    styled_df = df.style
    for col in subsets["low_good"]:
        if col in df.columns:
            styled_df = styled_df.applymap(lambda v: highlight_numeric(v, low_good=True), subset=[col])
    for col in subsets["high_good"]:
        if col in df.columns:
            styled_df = styled_df.applymap(lambda v: highlight_numeric(v, low_good=False), subset=[col])

    st.subheader("📊 השוואת נתונים בין הדגמים")
    st.dataframe(styled_df, use_container_width=True)

    csv = df.to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד כ-CSV", csv, "car_advisor.csv", "text/csv")

if "summary" in st.session_state:
    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<a href="https://infocar.co.il/" target="_blank">'
            f'<button style="background-color:#117A65;color:white;padding:10px 20px;'
            f'border:none;border-radius:8px;font-size:16px;cursor:pointer;">'
            f'🔗 בדוק עבר ביטוחי ב-InfoCar</button></a>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("🚗 רצוי לקחת את הרכב לבדיקה במכון בדיקה מורשה לפני רכישה.")

# =============================
# כפתור הורדה של כל היסטוריית השאלונים
# =============================
log_file = "car_advisor_logs.csv"
if os.path.exists(log_file):
    with open(log_file, "rb") as f:
        st.download_button(
            "⬇️ הורד את כל היסטוריית השאלונים",
            f,
            file_name="car_advisor_logs.csv",
            mime="text/csv"
        )
