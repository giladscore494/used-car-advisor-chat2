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
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not OPENAI_API_KEY or not FIRECRAWL_API_KEY:
    st.error("❌ לא נמצאו מפתחות API. ודא שהגדרת אותם ב-secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# GPT – בחירת 20 דגמים (ללא מחיר/טורבו)
# =============================
def fetch_models_with_gpt(answers):
    prompt = f"""
    המשתמש הגדיר את ההעדפות הבאות:
    {answers}

    החזר עד 20 דגמים שמתאימים לשוק הישראלי.

    כללים חשובים:
    - אם התקציב נמוך (עד 20 אלף ₪) → החזר רק רכבים ישנים, פשוטים, עם תחזוקה זולה.
    - אם התקציב בינוני (20–40 אלף ₪) → החזר רכבים משפחתיים משומשים ונפוצים.
    - אם התקציב גבוה (40–80 אלף ₪) → החזר רכבים משומשים חדשים יותר.
    - אם התקציב מעל 80 אלף ₪ → אפשר גם רכבים חדשים יחסית.
    - אסור בשום אופן להחזיר רכבים יקרים יותר מהתקציב בפועל.

    אסור להחזיר מחיר או מידע על טורבו – זה ייבדק בנפרד.

    החזר אך ורק JSON תקני במבנה:
    [
      {{
        "model": "שם דגם",
        "year_range": "שנות ייצור מתאימות",
        "engine": "סוג מנוע",
        "car_type": "סוג רכב",
        "insurance": "עלות ביטוח משוערת",
        "license_fee": "אגרת רישוי",
        "maintenance": "תחזוקה שנתית משוערת",
        "common_issues": "תקלות נפוצות",
        "fuel_consumption": "צריכת דלק משוערת",
        "depreciation": "ירידת ערך משוערת (%)",
        "safety": "דירוג בטיחות",
        "parts_availability": "זמינות חלפים",
        "resale_value": "שמירת ערך"
      }}
    ]
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    try:
        return json.loads(re.search(r"\[.*\]", answer, re.S).group())
    except Exception:
        return []

# =============================
# סקרייפר – מחזיר מחיר וטורבו
# =============================
def scrape_price_and_turbo_batch(models):
    url = "https://api.firecrawl.dev/v1/scrape"
    query = " OR ".join([m["model"] for m in models])
    payload = {"url": f"https://www.carzy.co.il/search?q={query}"}
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        data = r.json().get("text", "")
    except Exception as e:
        st.warning(f"⚠️ שגיאת סקרייפר: {e}")
        data = ""

    enriched = {}
    for m in models:
        name = m["model"]
        # Regex למציאת טווח מחירים
        match = re.search(r"(\d{2},\d{3})[-–](\d{2},\d{3})", data)
        price = f"{match.group(1)}–{match.group(2)} ₪" if match else None
        turbo = 1 if ("טורבו" in data or "TURBO" in data) else 0
        enriched[name] = (price, turbo)
    return enriched

# =============================
# אימותים
# =============================
def verify_model_in_mot(df, model_name):
    return any(df["model"].astype(str).str.contains(model_name, case=False, na=False))

def verify_budget(price_range, budget_min, budget_max):
    if not price_range:
        return False

    # טווח תקציב עם חריגה ±13%
    budget_min_eff = budget_min * 0.87
    budget_max_eff = budget_max * 1.13

    nums = [re.sub(r"[^\d]", "", x) for x in price_range.replace("–","-").split("-")]
    nums = [int(x) for x in nums if x.isdigit()]
    if len(nums) != 2:
        return False

    min_price, max_price = min(nums), max(nums)
    return not (max_price < budget_min_eff or min_price > budget_max_eff)

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
    # שלב 0: אילו קבצים קיימים
    st.write("📂 קבצים בתיקייה הנוכחית:", os.listdir("."))

    with st.spinner("🧠 GPT מחפש דגמים מתאימים..."):
        models = fetch_models_with_gpt(answers)
        st.write(f"🔎 GPT החזיר {len(models)} דגמים:")
        st.json(models)

    with st.spinner("🌐 סקרייפר בודק מחירים וטורבו..."):
        scraped_data = scrape_price_and_turbo_batch(models)
        st.write("📊 נתוני סקרייפר:")
        st.json(scraped_data)

    with st.spinner("✅ סינון קשיח..."):
        try:
            mot_df = pd.read_csv("car_models_israel_clean.csv")
        except FileNotFoundError as e:
            st.error("❌ קובץ car_models_israel_clean.csv לא נמצא. ודא שהוא באמת נמצא ב-GitHub ובאותה תיקייה של app.py")
            raise e

        final_models = []
        debug_log = []
        for m in models:
            model_name = m["model"]
            reason = []

            if not verify_model_in_mot(mot_df, model_name):
                reason.append("❌ לא נמצא במאגר משרד התחבורה")

            price, turbo_val = scraped_data.get(model_name, (None, None))
            if not verify_budget(price, answers["budget_min"], answers["budget_max"]):
                reason.append("❌ מחיר לא בתקציב (גם אחרי סטייה 13%)")

            if answers["turbo"] != "לא משנה":
                if (answers["turbo"] == "כן" and turbo_val == 0) or \
                   (answers["turbo"] == "לא" and turbo_val == 1):
                    reason.append("❌ לא עומד בדרישת טורבו")

            if not reason:
                m["price_range"] = price
                m["turbo"] = turbo_val
                final_models.append(m)
                debug_log.append({model_name: "✅ עבר"})
            else:
                debug_log.append({model_name: reason})

        st.write("📝 דוח סינון:")
        st.json(debug_log)

    if final_models:
        df = pd.DataFrame(final_models)
        st.subheader("📊 דגמים מתאימים")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ לא נמצאו דגמים מתאימים לתקציב והעדפות שלך.")