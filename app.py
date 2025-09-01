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
# עזר: סינון ראשוני מול מאגר
# =============================
def _safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def filter_with_mot(answers, mot_file="car_models_israel_clean.csv"):
    if not os.path.exists(mot_file):
        st.error(f"❌ קובץ המאגר '{mot_file}' לא נמצא בתיקייה. ודא שהעלית אותו.")
        return []

    df = pd.read_csv(mot_file)

    for col in ["year", "engine_cc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    year_min = int(answers["year_min"])
    year_max = int(answers["year_max"])
    cc_min = int(answers["engine_cc_min"])
    cc_max = int(answers["engine_cc_max"])

    mask_year = df["year"].between(year_min, year_max, inclusive="both")
    mask_cc = df["engine_cc"].between(cc_min, cc_max, inclusive="both")
    mask_fuel = df["fuel"].astype(str) == answers["engine"]
    mask_gear = (answers["gearbox"] == "לא משנה") | (
        (answers["gearbox"] == "אוטומט") & (df["automatic"] == 1)
    ) | (
        (answers["gearbox"] == "ידני") & (df["automatic"] == 0)
    )

    df_filtered = df[mask_year & mask_cc & mask_fuel & mask_gear].copy()
    return df_filtered.to_dict(orient="records")

# =============================
# שלב 2 – Gemini העשרה מלאה
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

כעת בצע העשרה מלאה לכל רכב:
- אסור להוסיף דגמים חדשים
- חובה להתייחס לשוק הרכב בישראל בלבד
- חובה להחזיר טווח מחירים שתואם אך ורק לתקציב: {answers['budget_min']}–{answers['budget_max']} ₪
- אם דגם לא עומד בתקציב – אל תחזיר אותו כלל
- אם אין רכבים מתאימים – החזר JSON ריק ({{}})

פורמט פלט נדרש – JSON בלבד:
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
     "parts_availability": "זמינות חלפים בישראל",
     "turbo": 0 או 1
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
# Cache – שמירה
# =============================
def save_cache(enriched_data, filename="cache.csv"):
    if not isinstance(enriched_data, dict) or not enriched_data:
        print("⚠️ enriched_data לא בפורמט dict – לא שומר לקובץ")
        return

    try:
        df_new = pd.DataFrame.from_dict(enriched_data, orient="index")
    except Exception as e:
        print(f"⚠️ שגיאה בשמירה ל־DataFrame: {e}")
        return

    if os.path.exists(filename):
        try:
            df_old = pd.read_csv(filename)
            df_final = pd.concat([df_old, df_new], axis=0)
        except:
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_csv(filename, index=True, encoding="utf-8-sig")

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

# =============================
# טיפול אחרי שליחה
# =============================
if submitted:
    with st.spinner("📊 סינון ראשוני מול מאגר משרד התחבורה..."):
        verified_models = filter_with_mot(answers)

    with st.spinner("🌐 Gemini מעשיר נתונים..."):
        enriched_data = fetch_models_10params(answers, verified_models)

    save_cache(enriched_data)

    try:
        df_params = pd.DataFrame.from_dict(enriched_data, orient="index")
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
            "parts_availability": "זמינות חלפים",
            "turbo": "טורבו"
        }
        df_params.rename(columns=COLUMN_TRANSLATIONS, inplace=True)

        st.session_state["df_params"] = df_params
        st.subheader("🟩 טבלת 10 פרמטרים")
        st.dataframe(df_params, use_container_width=True)

    except Exception as e:
        st.warning("⚠️ בעיה בנתוני JSON")
        st.write(enriched_data)

    with st.spinner("⚡ GPT מסכם ומדרג..."):
        summary = final_recommendation_with_gpt(answers, enriched_data)
        st.session_state["summary"] = summary

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])

    # שמירה להיסטוריה
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "answers": json.dumps(answers, ensure_ascii=False),
        "params_data": json.dumps(enriched_data, ensure_ascii=False),
        "summary": st.session_state["summary"],
    }
    log_file = "car_advisor_logs.csv"
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        new_df = pd.DataFrame([record])
        final = pd.concat([existing, new_df], ignore_index=True)
    else:
        final = pd.DataFrame([record])
    final.to_csv(log_file, index=False, encoding="utf-8-sig")

# =============================
# הורדת טבלה
# =============================
if "df_params" in st.session_state:
    csv2 = st.session_state["df_params"].to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד טבלת 10 פרמטרים", csv2, "params_data.csv", "text/csv")

if os.path.exists("car_advisor_logs.csv"):
    with open("car_advisor_logs.csv", "rb") as f:
        st.download_button("⬇️ הורד את כל היסטוריית השאלונים", f, file_name="car_advisor_logs.csv", mime="text/csv")
