
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
# עזר: זיהוי סוגי דלק/הנעה כפי שמופיעים במאגר
# =============================
def _safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# --- פונקציות חדשות לנרמול ---
def normalize_fuel_type(fuel_text: str) -> str:
    """
    מנרמל את סוג הדלק לפי ערכי המאגר.
    מחזיר אחד מתוך:
    'בנזין', 'דיזל', 'היברידי-בנזין', 'היברידי-דיזל', 'חשמלי'
    """
    if not fuel_text or pd.isna(fuel_text):
        return ""
    s = str(fuel_text).strip().lower()

    # חשמל מלא
    if s in ["חשמל", "חשמלי"] or "bev" in s or "battery" in s:
        return "חשמלי"

    # היברידי-בנזין
    if "חשמל/בנזין" in s or ("היברידי" in s and "דיזל" not in s) or "phev" in s or "plug" in s:
        return "היברידי-בנזין"

    # היברידי-דיזל
    if "חשמל/דיזל" in s or ("היברידי" in s and "דיזל" in s):
        return "היברידי-דיזל"

    # בנזין
    if "בנזין" in s or "petrol" in s or "gasoline" in s:
        return "בנזין"

    # דיזל
    if "דיזל" in s or "diesel" in s:
        return "דיזל"

    return fuel_text

def _is_hybrid_petrol(fuel_text: str) -> bool:
    return normalize_fuel_type(fuel_text) == "היברידי-בנזין"

def _is_hybrid_diesel(fuel_text: str) -> bool:
    return normalize_fuel_type(fuel_text) == "היברידי-דיזל"

def _is_hybrid(fuel_text: str) -> bool:
    """
    תאימות לאחור – מחזיר True לכל סוג היברידי (בנזין או דיזל).
    """
    return normalize_fuel_type(fuel_text) in ["היברידי-בנזין", "היברידי-דיזל"]
# --- סוף הוספה ---

def _is_electric(fuel_text: str) -> bool:
    """
    חשמלי מלא. מזהה 'חשמל' / 'חשמלי' / מונחים נפוצים ל-BEV.
    """
    s = _safe_str(fuel_text)
    if not s:
        return False
    s_low = s.lower()
    return s.strip() in ["חשמל", "חשמלי"] or any(k in s_low for k in ["bev", "battery electric"])

def _match_conventional(fuel_text: str, wanted: str) -> bool:
    """
    התאמה לבנזין/דיזל שאינם היברידיים (לא מכילים חשמל או מילות היבריד).
    """
    s = _safe_str(fuel_text)
    if not s:
        return False
    if _is_hybrid(s) or _is_electric(s):
        return False
    return wanted in s

def _engine_mask(df: pd.DataFrame, wanted_engine: str) -> pd.Series:
    """
    יוצר מסכה לוגית לעמודת fuel בהתאם לבחירת המשתמש
    """
    fuel_series = df["fuel"].astype(str).fillna("")
    if wanted_engine == "היברידי":
        return fuel_series.map(_is_hybrid)
    elif wanted_engine == "היברידי-בנזין":
        return fuel_series.map(_is_hybrid_petrol)
    elif wanted_engine == "היברידי-דיזל":
        return fuel_series.map(_is_hybrid_diesel)
    elif wanted_engine == "חשמלי":
        return fuel_series.map(_is_electric)
    elif wanted_engine in ["בנזין", "דיזל"]:
        return fuel_series.map(lambda s: _match_conventional(s, wanted_engine))
    else:
        return pd.Series([True] * len(df), index=df.index)

def _gearbox_mask(df: pd.DataFrame, wanted: str) -> pd.Series:
    if wanted == "לא משנה":
        return pd.Series([True] * len(df), index=df.index)
    elif wanted == "אוטומט":
        return df["automatic"].astype(int) == 1
    else:  # "ידני"
        return df["automatic"].astype(int) == 0

# =============================
# שלב 1 – סינון ראשוני מול מאגר משרד התחבורה
# =============================
def filter_with_mot(answers, mot_file="car_models_israel.csv"):
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
    mask_fuel = _engine_mask(df, answers["engine"])
    mask_gear = _gearbox_mask(df, answers["gearbox"])

    df_filtered = df[mask_year & mask_cc & mask_fuel & mask_gear].copy()
    return df_filtered.to_dict(orient="records")

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
                ✅ החזר אך ורק JSON תקני עם השדות...
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
        ["בנזין", "דיזל", "היברידי", "היברידי-בנזין", "היברידי-דיזל", "חשמלי"],
        help="במאגר הממשלתי 'היברידי' מופיע לרוב כ'בנזין/חשמל' או 'דיזל/חשמל'."
    )
    ...