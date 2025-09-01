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
# סינון ראשוני מול מאגר משרד התחבורה
# =============================
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

    df_filtered = df[mask_year & mask_cc].copy()
    return df_filtered.to_dict(orient="records")

# =============================
# Gemini בונה טבלת 10 פרמטרים
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

כעת בצע סינון משלים והחזר פרמטרים יבשים בלבד.

⚠️ תנאי קריטי:
החזר אך ורק דגמים שטווח המחירון שלהם נמצא בין {answers['budget_min']} ל-{answers['budget_max']} ₪.
אם טווח המחירים חורג ולו במעט – אל תחזיר את הדגם הזה.
אם אין רכבים מתאימים – החזר JSON ריק ({{}}).

פורמט פלט – JSON תקני בלבד עם השדות:
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
# GPT מסכם ומדרג
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
# שמירת קאש
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
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'
