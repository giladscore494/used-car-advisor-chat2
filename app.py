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
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
        cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        return {"error": str(e), "raw": cleaned}

# =============================
# שלב 1 – Gemini מחזיר רכבים בפורמט משרד התחבורה
# =============================
def fetch_models_from_mot_format(answers, mot_file="car_models_israel.csv"):
    mot_df = pd.read_csv(mot_file)
    sample = mot_df.head(20).to_dict(orient="records")

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                הנה דוגמה למבנה הטבלה שלי:
                {sample}

                המשתמש נתן את ההעדפות הבאות:
                {answers}

                החזר לפחות 10 דגמים מתאימים שנמכרים בישראל,
                אך ורק אם הם עומדים בקריטריונים:
                - מחיר {answers['budget_min']}–{answers['budget_max']} ₪
                - שנות ייצור {answers['year_range']}
                - סוג רכב {answers['car_type']}
                - סוג גיר {answers['gearbox']}
                - טווח נפח מנוע {answers['engine_cc_min']}–{answers['engine_cc_max']} סמ״ק
                - מנוע טורבו: {answers['turbo']}
                - מנוע מועדף: {answers['engine']}

                החזר JSON בפורמט זהה לדוגמה:
                רשימה של אובייקטים עם השדות:
                brand, model, year, engine_cc, fuel, automatic, turbo
                """
            }]
        }]
    }

    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 2 – Gemini מחזיר טבלת 10 פרמטרים
# =============================
def fetch_models_10params(verified_models):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                בנה טבלה של הדגמים הבאים בלבד:
                {', '.join([m['brand'] + ' ' + m['model'] for m in verified_models])}

                עבור כל דגם החזר JSON עם השדות:
                {{
                  "Model Name": {{
                     "price_range": "טווח מחירון ביד שנייה (₪)",
                     "availability": "זמינות בישראל",
                     "insurance_total": "עלות ביטוח חובה + צד ג' (₪)",
                     "license_fee": "אגרת רישוי/טסט שנתית (₪)",
                     "maintenance": "תחזוקה שנתית ממוצעת (₪)",
                     "common_issues": "תקלות נפוצות",
                     "fuel_consumption": "צריכת דלק אמיתית (ק״מ לליטר)",
                     "depreciation": "ירידת ערך ממוצעת (%)",
                     "safety": "דירוג בטיחות (כוכבים)",
                     "parts_availability": "זמינות חלפים בישראל"
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
# שלב 3 – GPT מסכם ומדרג
# =============================
def final_recommendation_with_gpt(answers, mot_data, params_data):
    text = f"""
    תשובות המשתמש:
    {answers}

    נתוני משרד התחבורה:
    {mot_data}

    נתוני 10 פרמטרים:
    {params_data}

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
def save_log(answers, mot_data, params_data, summary, filename="car_advisor_logs.csv"):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "answers": json.dumps(answers, ensure_ascii=False),
        "mot_data": json.dumps(mot_data, ensure_ascii=False),
        "params_data": json.dumps(params_data, ensure_ascii=False),
        "summary": summary,
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

with st.form("car_form"):
    answers = {}
    answers["budget_min"] = int(st.text_input("תקציב מינימלי (₪)", "5000"))
    answers["budget_max"] = int(st.text_input("תקציב מקסימלי (₪)", "20000"))
    answers["engine"] = st.radio("מנוע מועדף:", ["בנזין", "דיזל", "היברידי", "חשמלי"])
    answers["engine_cc_min"] = int(st.text_input("נפח מנוע מינימלי (סמ״ק):", "1200"))
    answers["engine_cc_max"] = int(st.text_input("נפח מנוע מקסימלי (סמ״ק):", "2000"))
    answers["turbo"] = st.radio("מנוע טורבו:", ["לא משנה", "כן", "לא"])
    answers["year_range"] = st.text_input("שנות ייצור (לדוגמה: 1970–2015):", "2010–2015")
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'בק", "SUV", "טנדר", "משפחתי"])
    answers["gearbox"] = st.radio("גיר:", ["לא משנה", "אוטומט", "ידני", "רובוטי"])
    answers["usage"] = st.radio("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    answers["driver_age"] = st.selectbox("גיל הנהג הראשי:", ["עד 21", "21–24", "25–34", "35+"])
    answers["license_years"] = st.selectbox("ותק רישיון נהיגה:", ["פחות משנה", "1–3 שנים", "3–5 שנים", "מעל 5 שנים"])
    answers["insurance_history"] = st.selectbox("עבר ביטוחי/תעבורתי:", ["ללא", "תאונה אחת", "מספר תביעות"])
    answers["maintenance_budget"] = st.selectbox("יכולת תחזוקה:", ["מתחת 3,000 ₪", "3,000–5,000 ₪", "מעל 5,000 ₪"])
    answers["extra"] = st.text_area("משהו נוסף שתרצה לציין?")

    submitted = st.form_submit_button("שלח וקבל המלצה")

# =============================
# טיפול אחרי שליחה
# =============================
if submitted:
    with st.spinner("🌐 Gemini מחפש רכבים בפורמט משרד התחבורה..."):
        mot_data = fetch_models_from_mot_format(answers)

    st.subheader("🟦 רכבים מאומתים (נתוני משרד התחבורה)")
    st.write(mot_data)

    with st.spinner("🌐 Gemini בונה טבלת 10 פרמטרים..."):
        params_data = fetch_models_10params(mot_data)

    try:
        df_params = pd.DataFrame(params_data).T
        st.subheader("🟩 טבלת 10 פרמטרים")
        st.dataframe(df_params, use_container_width=True)
    except Exception as e:
        st.warning("⚠️ בעיה בנתוני JSON")
        st.write(params_data)

    with st.spinner("⚡ GPT מסכם ומדרג..."):
        summary = final_recommendation_with_gpt(answers, mot_data, params_data)

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(summary)

    save_log(answers, mot_data, params_data, summary)

    csv1 = pd.DataFrame(mot_data).to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ הורד נתוני משרד התחבורה", csv1, "mot_data.csv", "text/csv")

    csv2 = df_params.to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד טבלת 10 פרמטרים", csv2, "params_data.csv", "text/csv")

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
# כפתור הורדה של כל ההיסטוריה
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
