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
# שלב 1 – Gemini מייצר עד 20 דגמים
# =============================
def gemini_propose_models(answers, max_retries=5, wait_seconds=2):
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"""
                המשתמש נתן את ההעדפות הבאות:
                {answers}

                המשימה שלך: הצע עד 20 דגמים שמתאימים לשאלון. 
                כל דגם חייב להיות מוחזר בפורמט JSON עם כל הפרמטרים.

                עבור כל דגם החזר JSON בפורמט:
                {{
                  "Model (year, engine, fuel)": {{
                     "price_range": "טווח מחירון ביד שנייה בישראל (₪)",
                     "availability": "זמינות בישראל",
                     "insurance_total": "עלות ביטוח חובה + צד ג' (₪)",
                     "license_fee": "אגרת רישוי/טסט שנתית (₪)",
                     "maintenance": "תחזוקה שנתית ממוצעת (₪)",
                     "common_issues": "תקלות נפוצות",
                     "fuel_consumption": "צריכת דלק אמיתית (ק״מ לליטר)",
                     "depreciation": "ירידת ערך ממוצעת (%)",
                     "safety": "דירוג בטיחות (כוכבים)",
                     "parts_availability": "זמינות חלפים בישראל",
                     "turbo": 0/1,
                     "status": "included/excluded",
                     "reason": "הסבר קצר למה נכלל או נפסל"
                  }}
                }}

                חוקים:
                - החזר לפחות 5 דגמים (ועד 20).
                - חובה להתחשב בכל ההעדפות שניתנו.
                - החזר מספרים בלבד בטווח המחיר (למשל: 25000-35000).
                - אסור להחזיר טקסט חופשי – רק JSON חוקי.
                """
            }]
        }]
    }

    for attempt in range(max_retries):
        answer = safe_gemini_call(payload)
        parsed = parse_gemini_json(answer)

        if parsed and isinstance(parsed, dict) and len(parsed) >= 1:
            return parsed

        time.sleep(wait_seconds)

    return {}

# =============================
# שלב 2 – הצלבה עם מאגר משרד התחבורה
# =============================
def cross_check_with_mot(gemini_models, mot_file="car_models_israel_clean.csv"):
    if not os.path.exists(mot_file):
        st.error(f"❌ קובץ המאגר '{mot_file}' לא נמצא בתיקייה.")
        return gemini_models

    df = pd.read_csv(mot_file)
    df_models = df["model"].astype(str).str.lower().unique().tolist()

    checked = {}
    for model, values in gemini_models.items():
        model_name = model.split("(")[0].strip().lower()
        if model_name in df_models:
            checked[model] = values
        else:
            values["status"] = "excluded"
            values["reason"] = "לא נמצא במאגר משרד התחבורה"
            checked[model] = values

    return checked

# =============================
# שלב 3 – Debug + סינון תקציב
# =============================
def debug_and_filter(params_data, budget_min, budget_max):
    results = {}
    lower_limit = budget_min * 0.9
    upper_limit = budget_max * 1.1

    st.subheader("🔎 Debug – בדיקת דגמים מול כל החוקים")
    st.write(f"גבולות תקציב לאחר סטייה: {lower_limit} – {upper_limit}")

    if not params_data:
        st.warning("⚠️ Gemini לא החזיר בכלל דגמים")
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
# שלב 4 – GPT מסכם ומדרג
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
    - התייחס לכל 10 הפרמטרים (ביטוח, רישוי, תחזוקה, אמינות, צריכת דלק, ירידת ערך וכו’)
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
    with st.spinner("🌐 Gemini מייצר עד 20 דגמים עם פרמטרים..."):
        gemini_models = gemini_propose_models(answers)

    with st.spinner("📊 הצלבה מול מאגר משרד התחבורה..."):
        checked_models = cross_check_with_mot(gemini_models)

    filtered_models = debug_and_filter(checked_models, answers["budget_min"], answers["budget_max"])
    if not filtered_models:
        st.warning("⚠️ לא נמצאו רכבים מתאימים")
        st.stop()

    with st.spinner("⚡ GPT מסכם ומדרג..."):
        summary = final_recommendation_with_gpt(answers, filtered_models)
        st.session_state["summary"] = summary

    st.subheader("🔎 ההמלצה הסופית שלך")
    st.write(st.session_state["summary"])

    save_log(answers, filtered_models, st.session_state["summary"])