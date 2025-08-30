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
        return []

# =============================
# שלב 1 – סינון ראשוני מול מאגר משרד התחבורה
# =============================
def filter_with_mot(answers, mot_file="car_models_israel.csv"):
    df = pd.read_csv(mot_file)

    df = df[
        (df["year"].between(int(answers["year_min"]), int(answers["year_max"]))) &
        (df["engine_cc"].between(int(answers["engine_cc_min"]), int(answers["engine_cc_max"]))) &
        (df["fuel"] == answers["engine"]) &
        ((answers["gearbox"] == "לא משנה") | (df["automatic"] == (1 if answers["gearbox"] == "אוטומט" else 0)))
    ]

    return df.to_dict(orient="records")

# =============================
# שלב 2 – Gemini בונה טבלת 10 פרמטרים (סינון משלים)
# =============================
def fetch_models_10params(answers, verified_models):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                הנה רשימת רכבים שעברו סינון ראשוני ממאגר משרד התחבורה:
                {verified_models}

                כעת בצע סינון משלים לפי כל ההעדפות:
                - סוג רכב: {answers['car_type']}
                - מנוע טורבו: {answers['turbo']}
                - שימוש עיקרי: {answers['usage']}
                - גיל נהג: {answers['driver_age']}
                - ותק רישיון: {answers['license_years']}
                - עבר ביטוחי: {answers['insurance_history']}
                - תקציב תחזוקה: {answers['maintenance_budget']}
                - אמינות מול נוחות: {answers['reliability_vs_comfort']}
                - שמירת ערך: {answers['resale_value']}
                - שיקולי איכות סביבה: {answers['eco_pref']}
                - תקציב כולל: {answers['budget_min']}–{answers['budget_max']} ₪

                חשוב:
                ❌ אל תחזיר שום דגם שלא עומד בכל הקריטריונים.
                ✅ החזר אך ורק JSON תקני עם השדות:
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
                """
            }]
        }]
    }
    answer = safe_gemini_call(payload)
    return parse_gemini_json(answer)

# =============================
# שלב 3 – GPT מסכם ומדרג
# =============================
def final_recommendation_with_gpt(answers, params_data):
    text = f"""
    תשובות המשתמש:
    {answers}

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
def save_log(answers, params_data, summary, filename="car_advisor_logs.csv"):
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "answers": json.dumps(answers, ensure_ascii=False),
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
    st.caption("התקציב מגדיר כמה כסף מוכן להשקיע – מונע הצעות יקרות מדי או זולות מדי.")

    answers["engine"] = st.radio(
        "מנוע מועדף:",
        ["בנזין", "דיזל", "היברידי", "חשמלי"],
        help="סוג המנוע משפיע על צריכת הדלק, תחזוקה ועלויות – בנזין זול לתחזוקה, דיזל חסכוני בנסיעות ארוכות, היברידי/חשמלי ירוקים יותר."
    )

    answers["engine_cc_min"] = int(st.text_input("נפח מנוע מינימלי (סמ״ק):", "1200"))
    answers["engine_cc_max"] = int(st.text_input("נפח מנוע מקסימלי (סמ״ק):", "2000"))
    st.caption("נפח מנוע קובע את עוצמת המנוע ואת צריכת הדלק – גדול יותר = חזק יותר אבל יקר יותר בתחזוקה ובביטוח.")

    answers["year_min"] = st.text_input("שנת ייצור מינימלית:", "2000")
    answers["year_max"] = st.text_input("שנת ייצור מקסימלית:", "2020")
    st.caption("שנת הייצור קובעת את גיל הרכב – משפיע על אמינות, ירידת ערך ועלויות ביטוח.")

    answers["car_type"] = st.selectbox(
        "סוג רכב:",
        ["סדאן", "האצ'בק", "SUV", "מיני", "סופר מיני", "סטיישן", "טנדר", "משפחתי"],
        help="סוג הרכב קובע את הגודל, מרחב הפנים ונוחות הנסיעה."
    )

    answers["gearbox"] = st.radio(
        "גיר:",
        ["לא משנה", "אוטומט", "ידני"],
        help="גיר אוטומט נוח לנהיגה בעיר, ידני זול יותר בתחזוקה ודלק."
    )

    answers["turbo"] = st.radio(
        "מנוע טורבו:",
        ["לא משנה", "כן", "לא"],
        help="מנוע עם טורבו חזק יותר, אבל דורש תחזוקה יקרה יותר – מתאים למי שמחפש ביצועים."
    )

    answers["usage"] = st.radio(
        "שימוש עיקרי:",
        ["עירוני", "בין-עירוני", "מעורב"],
        help="עירוני = קומפקטי וחסכוני, בין-עירוני = מנוע חזק יותר, מעורב = שילוב של שניהם."
    )

    answers["driver_age"] = st.selectbox(
        "גיל הנהג הראשי:",
        ["עד 21", "21–24", "25–34", "35+"],
        help="גיל הנהג משפיע ישירות על עלות הביטוח – נהגים צעירים משלמים יותר."
    )

    answers["license_years"] = st.selectbox(
        "ותק רישיון נהיגה:",
        ["פחות משנה", "1–3 שנים", "3–5 שנים", "מעל 5 שנים"],
        help="נהג חדש נחשב מסוכן יותר לחברות הביטוח – לכן העלויות גבוהות יותר."
    )

    answers["insurance_history"] = st.selectbox(
        "עבר ביטוחי/תעבורתי:",
        ["ללא", "תאונה אחת", "מספר תביעות"],
        help="עבר נקי = ביטוח זול יותר. תאונות/תביעות מעלות משמעותית את המחיר."
    )

    answers["annual_km"] = st.selectbox(
        "נסועה שנתית (ק״מ):",
        ["עד 10,000", "10,000–20,000", "20,000–30,000", "מעל 30,000"],
        help="מי שנוסע הרבה צריך רכב אמין וחסכוני יותר בתחזוקה ובדלק."
    )

    answers["passengers"] = st.selectbox(
        "מספר נוסעים עיקרי:",
        ["לרוב לבד", "2 אנשים", "3–5 נוסעים", "מעל 5"],
        help="משפיע על גודל הרכב – מיני מתאים ליחיד/זוג, משפחתי מתאים ל-4–5 נוסעים."
    )

    answers["maintenance_budget"] = st.selectbox(
        "יכולת תחזוקה:",
        ["מתחת 3,000 ₪", "3,000–5,000 ₪", "מעל 5,000 ₪"],
        help="כמה כסף מוכן להוציא בשנה על טיפולים ותיקונים."
    )

    answers["reliability_vs_comfort"] = st.selectbox(
        "מה חשוב יותר?",
        ["אמינות מעל הכול", "איזון אמינות ונוחות", "נוחות/ביצועים"],
        help="אמינות = פחות מוסך. נוחות/ביצועים = רכב מהנה אבל עלול לעלות יותר בתחזוקה."
    )

    answers["eco_pref"] = st.selectbox(
        "שיקולי איכות סביבה:",
        ["חשוב רכב ירוק/חסכוני", "לא משנה"],
        help="מאפשר עדיפות לרכב היברידי/חשמלי כדי לחסוך בדלק ולזהם פחות."
    )

    answers["resale_value"] = st.selectbox(
        "שמירת ערך עתידית:",
        ["חשוב לשמור על ערך", "פחות חשוב"],
        help="שמירת ערך חשובה למי שמתכנן למכור את הרכב בעוד כמה שנים."
    )

    answers["extra"] = st.text_area(
        "משהו נוסף שתרצה לציין?",
        help="כאן אפשר להוסיף דרישות מיוחדות – כמו צבע, מערכות בטיחות או גג נפתח."
    )

    submitted = st.form_submit_button("שלח וקבל המלצה")

# =============================
# טיפול אחרי שליחה
# =============================
if submitted:
    with st.spinner("📊 סינון ראשוני מול מאגר משרד התחבורה..."):
        verified_models = filter_with_mot(answers)

    with st.spinner("🌐 Gemini בונה טבלת 10 פרמטרים..."):
        params_data = fetch_models_10params(answers, verified_models)

    try:
        df_params = pd.DataFrame(params_data).T

        COLUMN_TRANSLATIONS = {
            "price_range": "טווח מחירון",
            "availability": "זמינות בישראל",
            "insurance_total": "ביטוח חובה + צד ג׳",
            "license_fee": "אגרת רישוי",
            "maintenance": "תחזוקה שנתית",
            "common_issues": "תקלות נפוצות",
            "fuel_consumption": "צריכת דלק (ק״מ לליטר)",
            "depreciation": "ירידת ערך (%)",
            "safety": "דירוג בטיחות (כוכבים)",
            "parts_availability": "זמינות חלפים"
        }
        df_params.rename(columns=COLUMN_TRANSLATIONS, inplace=True)

        st.session_state["df_params"] = df_params

        st.subheader("🟩 טבלת 10 פרמטרים")
        st.dataframe(df_params, use_container_width=True)

    except Exception as e:
        st.warning("⚠️ בעיה בנתוני JSON")
        st.write(params_data)

    with st.spinner("⚡ GPT מסכם ומדרג..."):
        summary = final_recommendation_with_gpt(answers, params_data)
        st.session_state["summary"] = summary

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])

    save_log(answers, params_data, st.session_state["summary"])

# =============================
# הורדת טבלה מה-session
# =============================
if "df_params" in st.session_state:
    csv2 = st.session_state["df_params"].to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד טבלת 10 פרמטרים", csv2, "params_data.csv", "text/csv")

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
