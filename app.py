
import os
import re
import json
import requests
import datetime
import time
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
# שלב 1 – סינון ראשוני מול מאגר משרד התחבורה
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
    mask_fuel = df["fuel"] == answers["engine"]
    mask_gear = (answers["gearbox"] == "לא משנה") | \
                ((answers["gearbox"] == "אוטומט") & (df["automatic"] == 1)) | \
                ((answers["gearbox"] == "ידני") & (df["automatic"] == 0))

    df_filtered = df[mask_year & mask_cc & mask_fuel & mask_gear].copy()

    return df_filtered.to_dict(orient="records")

# =============================
# שלב 2א – Gemini מחזיר טווחי מחירים + status + reason
# =============================
def fetch_price_ranges(answers, verified_models, max_retries=5, wait_seconds=2):
    limited_models = verified_models[:20]

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                רשימת דגמים ממאגר משרד התחבורה (עד 20):
                {limited_models}

                עליך לבחור מהרשימה רק את הדגמים שתואמים להעדפות המשתמש:
                - סוג רכב: {answers['car_type']}
                - שימוש עיקרי: {answers['usage']}
                - גיל נהג ראשי: {answers['driver_age']}
                - תחזוקה מקסימלית: {answers['maintenance_budget']}
                - מספר נוסעים: {answers['passengers']}
                - אמינות מול נוחות: {answers['reliability_vs_comfort']}
                - שיקולי איכות סביבה: {answers['eco_pref']}
                - שמירת ערך עתידית: {answers['resale_value']}
                - מנוע טורבו: {answers['turbo']}

                עבור כל דגם החזר JSON בפורמט:
                {{
                  "Model (year, engine, fuel)": {{
                     "price_range": "טווח מחירון ביד שנייה בישראל (₪)",
                     "status": "included/excluded",
                     "reason": "הסבר קצר למה נכלל או נפסל"
                  }}
                }}

                חוקים:
                - חובה להתחשב בכל ההעדפות שניתנו.
                - החזר לפחות 5 דגמים (או קרובים ביותר אם אין התאמה מלאה).
                - החזר מספרים בלבד (לדוגמה: 55000–75000).
                - אסור להחזיר טקסט חופשי – רק JSON חוקי.
                """
            }]
        }]
    }

    for attempt in range(max_retries):
        answer = safe_gemini_call(payload)
        parsed = parse_gemini_json(answer)

        if parsed and isinstance(parsed, dict) and len(parsed) >= 1:
            return parsed  # ✅ ברגע שקיבלנו JSON טוב – ממשיכים

        time.sleep(wait_seconds)

    return {}

# =============================
# שלב 2ב – Debug מפורט על כל דגם
# =============================
def debug_and_filter(params_data, budget_min, budget_max):
    results = {}
    lower_limit = budget_min * 0.9
    upper_limit = budget_max * 1.1

    st.subheader("🔎 Debug – בדיקת דגמים מול כל החוקים")
    st.write(f"גבולות תקציב לאחר סטייה: {lower_limit} – {upper_limit}")

    if not params_data:
        st.warning("⚠️ Gemini לא החזיר בכלל דגמים לסינון")
        return {}

    for model, values in params_data.items():
        price_text = str(values.get("price_range", "")).lower()
        status = values.get("status", "unknown")
        reason = values.get("reason", "")

        # חילוץ מספרים מהמחיר
        nums = []
        for match in re.findall(r"\d[\d,]*", price_text):
            try:
                nums.append(int(match.replace(",", "").replace("₪","")))
            except:
                pass

        if "אלף" in price_text:
            try:
                k = int(re.search(r"(\d+)", price_text).group(1))
                if k < 1000:
                    nums.append(k * 1000)
            except:
                pass

        if "k" in price_text:
            try:
                k = int(re.search(r"(\d+)", price_text).group(1))
                nums.append(k * 1000)
            except:
                pass

        nums = sorted(set(nums))

        # בדיקת תקציב
        in_budget = False
        chosen_val = None
        for n in nums:
            if lower_limit <= n <= upper_limit:
                in_budget = True
                chosen_val = n
                break

        if status == "included" and in_budget:
            results[model] = values
            results[model]["_calculated_price"] = chosen_val
            st.write(f"✅ {model} → נכלל | סיבה: {reason} | מחיר: {price_text} → זוהה: {nums} → נבחר {chosen_val}")
        else:
            st.write(f"❌ {model} → נפסל | סיבה: {reason} | מחיר: {price_text} → זוהה: {nums}")

    return results

# =============================
# שלב 3 – GPT מסכם ומדרג
# =============================
def final_recommendation_with_gpt(answers, params_data):
    text = f"""
    תשובות המשתמש:
    {answers}

    נתוני פרמטרים:
    {params_data}

    צור סיכום בעברית:
    - בחר עד 5 דגמים בלבד
    - פרט יתרונות וחסרונות
    - התייחס לעלות ביטוח, תחזוקה, ירידת ערך, אמינות ושימוש עיקרי
    - הסבר למה הדגמים הכי מתאימים
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

    answers["engine"] = st.radio("מנוע מועדף:", ["בנזין", "דיזל", "היברידי-בנזין", "היברידי-דיזל", "חשמל"])
    answers["engine_cc_min"] = int(st.text_input("נפח מנוע מינימלי (סמ״ק):", "1200"))
    answers["engine_cc_max"] = int(st.text_input("נפח מנוע מקסימלי (סמ״ק):", "2000"))
    answers["year_min"] = st.text_input("שנת ייצור מינימלית:", "2000")
    answers["year_max"] = st.text_input("שנת ייצור מקסימלית:", "2020")

    answers["car_type"] = st.selectbox("סוג רכב:", ["סדאן", "האצ'בק", "SUV", "מיני", "סטיישן", "טנדר", "משפחתי"])
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

    with st.spinner("🌐 Gemini מחזיר טווחי מחירים + סיבות..."):
        price_data = fetch_price_ranges(answers, verified_models)

    filtered_models = debug_and_filter(price_data, answers["budget_min"], answers["budget_max"])
    if not filtered_models:
        st.warning("⚠️ לא נמצאו רכבים מתאימים")
        st.stop()

    with st.spinner("⚡ GPT מסכם ומדרג..."):
        summary = final_recommendation_with_gpt(answers, filtered_models)
        st.session_state["summary"] = summary

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])

    save_log(answers, filtered_models, st.session_state["summary"])