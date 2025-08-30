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
def safe_gemini_call(payload):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    try:
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
        data = r.json()
        if "candidates" not in data:
            return f"שגיאת Gemini: {data}"
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"שגיאה: {e}"

# =============================
# שלב 1 – Gemini מייצר רשימת דגמים מתאימים
# =============================
def generate_car_candidates_with_gemini(answers):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                החזר רשימה של עד 7 דגמים מתאימים לרכישה בישראל בטווח התקציב {answers['budget_min']}–{answers['budget_max']} ₪.

                החזר JSON בלבד, לדוגמה:
                ["Toyota Corolla 2018", "Hyundai i30 2019", "Mazda 3 2017"]
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    try:
        return json.loads(answer)
    except Exception as e:
        return {"error": str(e), "raw": answer}

# =============================
# שלב 2 – Gemini מחפש מידע יבש לכל דגם
# =============================
def fetch_models_data_with_gemini(models_list):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                מצא מידע יבש ברשת על הדגמים הבאים:
                {models_list}

                עבור כל דגם החזר בפורמט JSON:
                {{
                  "Model Name": {{
                     "price_range": "טווח מחירון ממוצע ביד שנייה",
                     "availability": "זמינות בישראל",
                     "insurance": "עלות ביטוח ממוצעת",
                     "license_fee": "אגרת רישוי/טסט שנתית",
                     "maintenance": "תחזוקה שנתית ממוצעת",
                     "common_issues": "תקלות נפוצות",
                     "fuel_consumption": "צריכת דלק אמיתית",
                     "depreciation": "ירידת ערך ממוצעת",
                     "safety": "דירוג בטיחות",
                     "parts_availability": "זמינות חלפים בישראל"
                  }}
                }}

                אל תוסיף טקסט מעבר ל-JSON.
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    try:
        match = re.search(r"\{.*\}", answer, re.S)
        if match:
            return json.loads(match.group(0))
        else:
            return {"error": "לא נמצא JSON", "raw": answer}
    except Exception as e:
        return {"error": str(e), "raw": answer}

# =============================
# שלב 3 – GPT מסנן ומסכם
# =============================
def final_recommendation_with_gpt(answers, models_data):
    text = f"""
    תשובות המשתמש:
    {answers}

    נתוני הדגמים:
    {models_data}

    צור המלצה בעברית:
    - בחר עד 5 דגמים מובילים
    - הצג טבלה עם כל הפרמטרים (מחירון, ביטוח, רישוי, תחזוקה, תקלות, דלק, ירידת ערך, בטיחות, חלפים)
    - הסבר יתרונות וחסרונות של כל דגם
    - נתח התאמה אישית לפי התקציב, מנוע, שנות ייצור, נוחות, חסכוניות
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

if submitted:
    with st.spinner("🤖 Gemini מחפש דגמים מתאימים..."):
        models_list = generate_car_candidates_with_gemini(answers)

    with st.spinner("🌐 Gemini בודק מידע יבש על הדגמים..."):
        models_data = fetch_models_data_with_gemini(models_list)

    try:
        df = pd.DataFrame(models_data).T
        df.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
        st.subheader("📊 השוואת נתונים בין הדגמים")
        st.dataframe(df, use_container_width=True)

        # כפתור הורדה ל-CSV
        csv = df.to_csv(index=True, encoding="utf-8-sig")
        st.download_button("⬇️ הורד כ-CSV", csv, "car_advisor.csv", "text/csv")

    except:
        st.warning("⚠️ בעיה בנתוני JSON")
        st.write(models_data)

    with st.spinner("⚡ GPT מסנן ומסכם..."):
        summary = final_recommendation_with_gpt(answers, models_data)

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(summary)

    # הערות חשובות
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
