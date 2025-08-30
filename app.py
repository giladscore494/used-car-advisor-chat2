import os
import re
import json
import requests
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
    # ננקה סימוני Markdown
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
        cleaned = cleaned.replace("```", "").strip()

    try:
        data = json.loads(cleaned)
        # אם זה מערך → נהפוך ל־dict מאוחד
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
# שלב 1 – Gemini מחזיר לפחות 10 דגמים עם טווח מחירים
# =============================
def fetch_models_data_with_gemini(answers):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                החזר לפחות 10 דגמים מתאימים לרכישה בישראל,
                אך ורק אם מחירם ביד שנייה נופל בטווח התקציב {answers['budget_min']}–{answers['budget_max']} ₪.
                אם יש יותר מ-10 אפשריים – בחר את ה-10 הטובים ביותר לפי ניתוח של אמינות, עלות ביטוח, תחזוקה, ירידת ערך ובטיחות.
                אם יש פחות מ-10 – החזר את כולם.

                עבור כל דגם החזר JSON תקני בלבד (במרכאות כפולות) עם השדות:
                {{
                  "Model Name": {{
                     "price_range": "טווח מחירון אמיתי ביד שנייה (₪, לדוגמה 6,000–8,000)",
                     "availability": "זמינות בישראל",
                     "insurance": "עלות ביטוח חובה + צד ג' ממוצעת (₪ לשנה, אמין)",
                     "license_fee": "אגרת רישוי/טסט שנתית (₪, לפי נפח מנוע)",
                     "maintenance": "תחזוקה שנתית ממוצעת (₪)",
                     "common_issues": "תקלות נפוצות",
                     "fuel_consumption": "צריכת דלק אמיתית (ק״מ לליטר)",
                     "depreciation": "ירידת ערך ממוצעת (%)",
                     "safety": "דירוג בטיחות (כוכבים)",
                     "parts_availability": "זמינות חלפים בישראל"
                  }}
                }}

                חובה:
                - החזר מינימום 10 דגמים אם קיימים.
                - החזר טווח מחירים אמיתי ולא מספר אחד.
                - אל תחרוג מהתקציב הנתון.
                - אל תמציא מספרים. אם מידע לא קיים – כתוב "לא נמצא".
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
# Streamlit UI
# =============================
st.set_page_config(page_title="Car-Advisor", page_icon="🚗")
st.title("🚗 Car-Advisor – יועץ רכבים חכם")

COLUMN_TRANSLATIONS = {
    "price_range": "טווח מחירון",
    "availability": "זמינות בישראל",
    "insurance": "עלות ביטוח",
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

    styled_df = df.style.applymap(lambda v: highlight_numeric(v, low_good=True), subset=["עלות ביטוח", "תחזוקה שנתית"])\
                        .applymap(lambda v: highlight_numeric(v, low_good=False), subset=["צריכת דלק"])

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
