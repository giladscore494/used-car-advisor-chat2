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
# עזר: זיהוי סוגי דלק/הנעה
# =============================
def _safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def normalize_fuel_type(fuel_text: str) -> str:
    if not fuel_text or pd.isna(fuel_text):
        return ""
    s = str(fuel_text).strip().lower()
    if s in ["חשמל", "חשמלי"] or "bev" in s or "battery" in s:
        return "חשמלי"
    if "חשמל/בנזין" in s or ("היברידי" in s and "דיזל" not in s) or "phev" in s or "plug" in s:
        return "היברידי-בנזין"
    if "חשמל/דיזל" in s or ("היברידי" in s and "דיזל" in s):
        return "היברידי-דיזל"
    if "בנזין" in s or "petrol" in s or "gasoline" in s:
        return "בנזין"
    if "דיזל" in s or "diesel" in s:
        return "דיזל"
    return fuel_text

def _is_hybrid_petrol(fuel_text: str) -> bool:
    return normalize_fuel_type(fuel_text) == "היברידי-בנזין"

def _is_hybrid_diesel(fuel_text: str) -> bool:
    return normalize_fuel_type(fuel_text) == "היברידי-דיזל"

def _is_hybrid(fuel_text: str) -> bool:
    return normalize_fuel_type(fuel_text) in ["היברידי-בנזין", "היברידי-דיזל"]

def _is_electric(fuel_text: str) -> bool:
    s = _safe_str(fuel_text)
    if not s:
        return False
    s_low = s.lower()
    return s.strip() in ["חשמל", "חשמלי"] or any(k in s_low for k in ["bev", "battery electric"])

def _match_conventional(fuel_text: str, wanted: str) -> bool:
    s = _safe_str(fuel_text)
    if not s:
        return False
    if _is_hybrid(s) or _is_electric(s):
        return False
    return wanted in s

def _engine_mask(df: pd.DataFrame, wanted_engine: str) -> pd.Series:
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
    else:
        return df["automatic"].astype(int) == 0

# =============================
# סינון ראשוני מול המאגר
# =============================
def filter_with_mot(answers, mot_file="car_models_israel.csv"):
    if not os.path.exists(mot_file):
        st.error(f"❌ קובץ המאגר '{mot_file}' לא נמצא בתיקייה.")
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
# אימות מחיר משופר
# =============================
def parse_price_range(txt: str):
    if not txt or not isinstance(txt, str):
        return None, None
    txt = txt.lower().replace(",", "").replace("₪", "").replace("ש״ח", "").replace("שח", "")
    txt = txt.replace("-", " ").replace("–", " ")
    nums = []
    for token in txt.split():
        if token.isdigit():
            nums.append(int(token))
        elif "אלף" in token:
            try:
                val = int(re.sub(r"\D", "", token)) * 1000
                nums.append(val)
            except:
                pass
        elif token.endswith("k"):
            try:
                val = int(re.sub(r"\D", "", token)) * 1000
                nums.append(val)
            except:
                pass
        else:
            try:
                val = int(re.sub(r"\D", "", token))
                if val > 0:
                    nums.append(val)
            except:
                pass
    if len(nums) >= 2:
        return min(nums), max(nums)
    elif len(nums) == 1:
        return nums[0], nums[0]
    return None, None

def filter_by_budget(df, budget_min, budget_max):
    def _row_in_budget(row):
        pmin, pmax = parse_price_range(str(row.get("טווח מחירון", "")))
        if pmin is None or pmax is None:
            return False
        # ✅ תנאי חדש: מספיק שיש חפיפה בין טווחי התקציב לטווח המחיר
        return not (pmax < budget_min or pmin > budget_max)
    return df[df.apply(_row_in_budget, axis=1)].copy()

# =============================
# Gemini – פרומפט נפרד להיברידי/חשמלי
# =============================
def fetch_models_10params(answers, verified_models):
    if answers["engine"] in ["היברידי", "היברידי-בנזין", "היברידי-דיזל", "חשמלי"]:
        if not verified_models:
            return {}
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": f"""
                    המשתמש נתן את ההעדפות הבאות:
                    {answers}

                    הנה רשימת רכבים שעברו סינון ראשוני ממאגר משרד התחבורה:
                    {verified_models}

                    ❌ מותר לבחור רק מתוך הרשימה.
                    ❌ אסור להמציא טווחי מחיר או דגמים.
                    ✅ אם אין דגמים מתאימים לתקציב – החזר JSON ריק: {{}}
                    """
                }]
            }]
        }
        answer = safe_gemini_call(payload)
        result = parse_gemini_json(answer)
        try:
            df_check = pd.DataFrame(result).T
            st.write("✅ DEBUG: לפני סינון תקציב", df_check.get("price_range"))
            df_check.rename(columns={"price_range": "טווח מחירון"}, inplace=True)
            df_check = filter_by_budget(df_check, int(answers["budget_min"]), int(answers["budget_max"]))
            st.write("✅ DEBUG: אחרי סינון תקציב", df_check.get("טווח מחירון"))
            if df_check.empty:
                return {}
            else:
                return result
        except Exception as e:
            st.write("❌ DEBUG Exception:", e)
            return {}
    else:
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": f"""
                    המשתמש נתן את ההעדפות הבאות:
                    {answers}

                    הנה רשימת רכבים שעברו סינון ראשוני ממאגר משרד התחבורה:
                    {verified_models}

                    ❌ אל תחזיר שום דגם שלא עומד בקריטריונים.
                    ✅ החזר אך ורק JSON תקני.
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
    answers["engine"] = st.radio(
        "מנוע מועדף:",
        ["בנזין", "דיזל", "היברידי", "היברידי-בנזין", "היברידי-דיזל", "חשמלי"],
    )
    answers["engine_cc_min"] = int(st.text_input("נפח מנוע מינימלי (סמ״ק):", "1200"))
    answers["engine_cc_max"] = int(st.text_input("נפח מנוע מקסימלי (סמ״ק):", "2000"))
    answers["year_min"] = st.text_input("שנת ייצור מינימלית:", "2000")
    answers["year_max"] = st.text_input("שנת ייצור מקסימלית:", "2020")
    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן","האצ'בק","SUV","מיני","סופר מיני","סטיישן","טנדר","משפחתי"])
    answers["gearbox"] = st.radio("גיר:", ["לא משנה", "אוטומט", "ידני"])
    answers["turbo"] = st.radio("מנוע טורבו:", ["לא משנה", "כן", "לא"])
    answers["usage"] = st.radio("שימוש עיקרי:", ["עירוני", "בין-עירוני", "מעורב"])
    answers["driver_age"] = st.selectbox("גיל הנהג הראשי:", ["עד 21","21–24","25–34","35+"])
    answers["license_years"] = st.selectbox("ותק רישיון נהיגה:", ["פחות משנה","1–3 שנים","3–5 שנים","מעל 5 שנים"])
    answers["insurance_history"] = st.selectbox("עבר ביטוחי/תעבורתי:", ["ללא","תאונה אחת","מספר תביעות"])
    answers["annual_km"] = st.selectbox("נסועה שנתית (ק״מ):", ["עד 10,000","10,000–20,000","20,000–30,000","מעל 30,000"])
    answers["passengers"] = st.selectbox("מספר נוסעים עיקרי:", ["לרוב לבד","2 אנשים","3–5 נוסעים","מעל 5"])
    answers["maintenance_budget"] = st.selectbox("יכולת תחזוקה:", ["מתחת 3,000 ₪","3,000–5,000 ₪","מעל 5,000 ₪"])
    answers["reliability_vs_comfort"] = st.selectbox("מה חשוב יותר?", ["אמינות מעל הכול","איזון אמינות ונוחות","נוחות/ביצועים"])
    answers["eco_pref"] = st.selectbox("שיקולי איכות סביבה:", ["חשוב רכב ירוק/חסכוני","לא משנה"])
    answers["resale_value"] = st.selectbox("שמירת ערך עתידית:", ["חשוב לשמור על ערך","פחות חשוב"])
    answers["extra"] = st.text_area("משהו נוסף שתרצה לציין?")
    submitted = st.form_submit_button("שלח וקבל המלצה")

if submitted:
    verified_models = filter_with_mot(answers)
    if not verified_models:
        st.warning("❌ לא נמצאו רכבים מתאימים במאגר.")
        st.stop()

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

        if answers["engine"] in ["היברידי","היברידי-בנזין","היברידי-דיזל","חשמלי"]:
            st.write("✅ DEBUG: לפני סינון תקציב", df_params[["טווח מחירון"]])
            df_params = filter_by_budget(df_params, int(answers["budget_min"]), int(answers["budget_max"]))
            st.write("✅ DEBUG: אחרי סינון תקציב", df_params[["טווח מחירון"]])

            if df_params.empty:
                st.warning("❌ לא נמצאו רכבים היברידיים/חשמליים בתקציב שהוזן.")
                st.stop()

        st.session_state["df_params"] = df_params
        st.subheader("🟩 טבלת 10 פרמטרים")
        st.dataframe(df_params, use_container_width=True)
    except Exception as e:
        st.warning("⚠️ בעיה בנתוני JSON")
        st.write(params_data)

    summary = final_recommendation_with_gpt(answers, params_data)
    st.session_state["summary"] = summary
    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])
    save_log(answers, params_data, summary)

if "df_params" in st.session_state:
    csv2 = st.session_state["df_params"].to_csv(index=True, encoding="utf-8-sig")
    st.download_button("⬇️ הורד טבלת 10 פרמטרים", csv2, "params_data.csv", "text/csv")