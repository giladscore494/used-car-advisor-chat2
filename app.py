# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – Streamlit Wizard (Modern Blue UI)
# All original logic preserved 1:1; questionnaire split into 5 steps
# Explanations in Hebrew; images removed; no duplicate fuel/energy columns
# =========================================

import streamlit as st
import pandas as pd
import json, os, uuid
from datetime import datetime
import numpy as np
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------------------- THEME (UI only) --------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rubik:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root { --primary:#2259b4; --accent:#ff7a18; --ink:#0f172a; --sub:#64748b; }
html, body, [class*="css"] { font-family: 'Inter','Rubik',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial !important; }
h1,h2,h3 { color: var(--ink) }
.btn-primary { background:var(--primary); color:#fff; padding:8px 16px; border-radius:10px; text-decoration:none; display:inline-block; }
.btn-ghost { background:#eef2ff; color:#1e293b; padding:8px 16px; border-radius:10px; }
.step { background:#fff; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,.06); padding:16px 18px; }
.pill { display:inline-block; background:#eef2ff; color:#273c75; border-radius:9999px; padding:2px 10px; font-weight:600; margin-right:6px;}
.disclaimer { color:#a16207; background:#fffbeb; border:1px solid #fde68a; padding:8px 12px; border-radius:10px; }
.card { border:1px solid #e5e7eb; border-radius:14px; padding:10px; }
.logo { height: 42px; margin-right:8px; vertical-align:middle; }
.topbar { display:flex; align-items:center; gap:10px; }
.small-muted { color:#64748b; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers (original logic) --------------------
def init_state():
    for key in [
        "user_profile","validated_cars","methods_info",
        "fuel_price","electricity_price","ui_step",
        "results_df","gemini_raw","search_info"
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.ui_step is None:
        st.session_state.ui_step = 0  # 0: start, 1..5 steps

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age,
                      family_size, cargo_need, safety_required,
                      trim_level, weights, body_style, driving_style, excluded_colors):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
        "family_size": family_size,
        "cargo_need": cargo_need,
        "safety_required": safety_required,
        "trim_level": trim_level,
        "weights": weights,
        "body_style": body_style,
        "driving_style": driving_style,
        "excluded_colors": excluded_colors,
    }

def clean_gemini_output(cars_raw):
    records, methods = [], []
    for car in cars_raw:
        if not isinstance(car, dict):
            continue
        record, method = {}, {}
        for k, v in car.items():
            if k.endswith("_method"):
                method[k] = v
            else:
                record[k] = v
        records.append(record)
        methods.append(method)
    return pd.DataFrame(records), methods

def normalize_car_values(df):
    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].replace({
            "בנזין": "gasoline",
            "דיזל": "diesel",
            "היברידי": "hybrid",
            "דיזל היברידי": "hybrid-diesel",
            "חשמלי": "electric"
        })
    if "gear" in df.columns:
        df["gear"] = df["gear"].replace({
            "אוטומטי": "automatic",
            "אוטומטית": "automatic",
            "אוטומטי (DSG7)": "automatic",
            "אוטומטי (TCT)": "automatic",
            "אוטומטי (רובוטי)": "automatic",
            "ידני": "manual",
            "ידנית": "manual"
        })
    if "turbo" in df.columns:
        df["turbo"] = df["turbo"].replace({
            "כן": True,
            "לא": False,
            True: True,
            False: False
        })
    return df

# -------- מיפויים (זהים) --------
fuel_map = {
    "בנזין": "gasoline",
    "היברידי": "hybrid",
    "דיזל היברידי": "hybrid-diesel",
    "דיזל": "diesel",
    "חשמלי": "electric"
}
gear_map = {"אוטומטית": "automatic", "ידנית": "manual"}
turbo_map = {"לא משנה": "any", "כן": "yes", "לא": "no"}

fuel_map_he = {v: k for k, v in fuel_map.items()}
gear_map_he = {v: k for k, v in gear_map.items()}
turbo_map_he = {"yes": "כן", "no": "לא", "any": "לא משנה", True: "כן", False: "לא"}

column_map_he = {
    "brand": "מותג",
    "model": "דגם",
    "year": "שנה",
    "fuel": "דלק",
    "gear": "תיבה",
    "turbo": "טורבו",
    "engine_cc": "נפח מנוע (סמ\"ק)",
    "price_range_nis": "טווח מחיר (₪)",
    "avg_fuel_consumption": "צריכת דלק ממוצעת (ק\"מ/ל')",  # יתעדכן אם electric
    "annual_fee": "אגרה שנתית (₪)",
    "annual_energy_cost": "עלות אנרגיה שנתית (₪)",
    "total_annual_cost": "עלות כוללת שנתית (₪)",
    "reliability_score": "אמינות",
    "maintenance_cost": "עלות אחזקה (₪/שנה)",
    "safety_rating": "בטיחות",
    "insurance_cost": "עלות ביטוח (₪/שנה)",
    "resale_value": "שמירת ערך",
    "performance_score": "ביצועים",
    "comfort_features": "נוחות",
    "suitability": "התאמה",
    "market_supply": "היצע בשוק"
}

method_map_he = {
    "fuel_method": "שיטת חישוב צריכת דלק/חשמל",
    "fee_method": "שיטת חישוב אגרה",
    "reliability_method": "שיטת חישוב אמינות",
    "maintenance_method": "שיטת חישוב עלות אחזקה",
    "safety_method": "שיטת חישוב בטיחות",
    "insurance_method": "שיטת חישוב ביטוח",
    "resale_method": "שיטת חישוב שמירת ערך",
    "performance_method": "שיטת חישוב ביצועים",
    "comfort_method": "שיטת חישוב נוחות",
    "suitability_method": "שיטת חישוב התאמה",
    "supply_method": "שיטת קביעת היצע"
}

# -------------------- UI Header --------------------
def topbar():
    st.markdown(
        '<div class="topbar">'
        '<img src="https://em-content.zobj.net/source/microsoft-teams/363/automobile_1f697.png" class="logo"/>'
        '<div><div style="font-weight:700;color:#0f172a;font-size:22px;">Car Advisor</div>'
        '<div style="color:#64748b;font-size:13px;">ייעוץ רכב | Wizard</div></div>'
        '<span class="pill">Modern</span><span class="pill">Fast</span><span class="pill">Clean UX</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

# -------------------- Init --------------------
init_state()
topbar()

# -------------------- Wizard Nav (unique keys) --------------------
def nav_buttons(left_label="חזור", right_label="הבא",
                left_action=None, right_action=None,
                show_left=True, show_right=True, extra=None):
    c1, c2 = st.columns([1,1])
    with c1:
        if show_left:
            st.button(left_label, on_click=left_action, use_container_width=False,
                      key=f"back_{st.session_state.ui_step}_{uuid.uuid4().hex}", type="secondary")
    with c2:
        if extra:
            extra()
        if show_right:
            st.button(right_label, on_click=right_action, use_container_width=False,
                      key=f"next_{st.session_state.ui_step}_{uuid.uuid4().hex}", type="primary")

# -------------------- STEP 0: START --------------------
if st.session_state.ui_step == 0:
    with st.container():
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.subheader("ברוך הבא ל-Car Advisor")
        st.write("מצא את הרכב הנכון עבורך ב-5 צעדים פשוטים.")
        st.markdown('</div>', unsafe_allow_html=True)

    def go_next(): st.session_state.ui_step = 1
    nav_buttons(show_left=False, right_label="התחל", right_action=go_next)

# -------------------- STEP 1: בסיס (מחיר/שנה/דלק/גיר/טורבו) --------------------
if st.session_state.ui_step == 1:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 1: בסיס")

    col1, col2, col3 = st.columns([1,1,1])
    with col1: budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
    with col2: budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
    with col3:
        ymin, ymax = st.columns(2)
        with ymin: year_min = st.number_input("שנתון מינימום", min_value=1990, max_value=datetime.now().year, value=2015)
        with ymax: year_max = st.number_input("שנתון מקסימום", min_value=1990, max_value=datetime.now().year, value=2019)

    fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
    if "חשמלי" in fuels_he:
        st.info("נבחר דלק חשמלי — תיבת ההילוכים נקבעת אוטומטית ל'אוטומטית'.")
        gears_he = ["אוטומטית"]
    else:
        gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])

    turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)

    # שמירה זמנית ב-state של שלב 1
    st.session_state._step1 = dict(
        budget_min=budget_min, budget_max=budget_max,
        year_min=year_min, year_max=year_max,
        fuels_he=fuels_he, gears_he=gears_he, turbo_choice_he=turbo_choice_he
    )

    st.markdown('</div>', unsafe_allow_html=True)

    def go_back(): st.session_state.ui_step = 0
    def go_next(): st.session_state.ui_step = 2
    nav_buttons(left_action=go_back, right_action=go_next)

# -------------------- STEP 2: שימוש וסגנון --------------------
if st.session_state.ui_step == 2:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 2: שימוש וסגנון")

    c4, c5, c6 = st.columns([2,1,1])
    with c4:
        main_use = st.text_area("תיאור הרכב והשימוש בו", value="נסיעה יומיומית לעבודה וטיולים קצרים", height=100)
    with c5:
        annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
    with c6:
        driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)

    c6a, c6b = st.columns(2)
    with c6a: license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
    with c6b: driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])

    cstyle1, cstyle2, cseats = st.columns([1,1,1])
    with cstyle1: body_style = st.selectbox("סגנון מרכב מועדף", ["כללי","סדאן","האצ'בק","קרוסאובר/ג'יפון"])
    with cstyle2: driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח","דינמי וספורטיבי"])
    with cseats: seats_choice = st.selectbox("מספר מקומות", ["4","5","5+"] )

    excluded_colors = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="").split(",")

    # שמירה זמנית של שלב 2
    st.session_state._step2 = dict(
        main_use=main_use, annual_km=annual_km, driver_age=driver_age,
        license_years=license_years, driver_gender=driver_gender,
        body_style=body_style, driving_style=driving_style,
        seats_choice=seats_choice, excluded_colors=excluded_colors
    )

    st.markdown('</div>', unsafe_allow_html=True)

    def go_back(): st.session_state.ui_step = 1
    def go_next(): st.session_state.ui_step = 3
    nav_buttons(left_action=go_back, right_action=go_next)

# -------------------- STEP 3: סדר עדיפויות --------------------
if st.session_state.ui_step == 3:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 3: סדר עדיפויות")

    st.markdown("#### (1–5)")
    reliability_weight = st.slider("אמינות", 1, 5, 5)
    resale_weight = st.slider("שמירת ערך", 1, 5, 3)
    fuel_weight = st.slider("חיסכון בדלק", 1, 5, 4)
    performance_weight = st.slider("ביצועים", 1, 5, 2)
    comfort_weight = st.slider("נוחות", 1, 5, 3)

    weights = {
        "reliability": reliability_weight,
        "resale": resale_weight,
        "fuel": fuel_weight,
        "performance": performance_weight,
        "comfort": comfort_weight,
    }
    st.session_state._step3 = dict(weights=weights)

    st.markdown('</div>', unsafe_allow_html=True)

    def go_back(): st.session_state.ui_step = 2
    def go_next(): st.session_state.ui_step = 4
    nav_buttons(left_action=go_back, right_action=go_next)

# -------------------- STEP 4: פרטים אישיים + היצע + מחירי אנרגיה --------------------
if st.session_state.ui_step == 4:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 4: פרטים נוספים")

    insurance_history = st.text_input("עבר ביטוחי", value="שנתיים ללא תביעות")
    violations = st.selectbox("דוחות/שלילות", ["אין", "שלילה בעבר", "נקודות פעילות"])

    cfam, ccargo, csafety, ctrim = st.columns([1,1,1,1])
    with cfam: family_size = st.selectbox("גודל משפחה", ["1-2","3-4","5+"])
    with ccargo: cargo_need = st.selectbox("צורך בתא מטען", ["קטן","בינוני","גדול"])
    with csafety: safety_required = st.radio("חובה מערכות בטיחות אקטיביות?", ["כן","לא"])
    with ctrim: trim_level = st.selectbox("רמת אבזור", ["בסיסי","סטנדרטי","עשיר"])

    consider_supply = st.radio("האם להתחשב בהיצע בשוק?", ["כן","לא"], index=0)

    cfp, cep = st.columns([1,1])
    with cfp: fuel_price = st.number_input("מחיר ליטר דלק (₪)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    with cep: electricity_price = st.number_input("מחיר חשמל לקוט״ש (₪)", min_value=0.1, max_value=5.0, value=0.65, step=0.01)

    st.session_state._step4 = dict(
        insurance_history=insurance_history, violations=violations,
        family_size=family_size, cargo_need=cargo_need, safety_required=safety_required,
        trim_level=trim_level, consider_supply=consider_supply,
        fuel_price=fuel_price, electricity_price=electricity_price
    )

    st.markdown('</div>', unsafe_allow_html=True)

    def go_back(): st.session_state.ui_step = 3
    def go_next(): st.session_state.ui_step = 5
    nav_buttons(left_action=go_back, right_action=go_next, right_label="המשך לייעוץ")

# -------------------- STEP 5: ייעוץ (Gemini) + תוצאות --------------------
if st.session_state.ui_step == 5:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 5: קבלת ייעוץ ותוצאות")

    # בניית הפרופיל (כמו במקור, עם תוספות אחרי היצירה)
    s1, s2, s3, s4 = st.session_state._step1, st.session_state._step2, st.session_state._step3, st.session_state._step4
    if not all([s1, s2, s3, s4]):
        st.error("חסרים נתונים בשלבים קודמים. חזור אחורה והשלם.")
    else:
        fuels = [fuel_map[f] for f in (s1["fuels_he"] or [])]
        gears = [gear_map[g] for g in (s1["gears_he"] or [])]
        turbo_choice = turbo_map[s1["turbo_choice_he"]]
        weights = s3["weights"]

        profile = make_user_profile(
            s1["budget_min"], s1["budget_max"], [s1["year_min"], s1["year_max"]],
            fuels, gears, turbo_choice, s2["main_use"], s2["annual_km"], s2["driver_age"],
            s4["family_size"], s4["cargo_need"], s4["safety_required"], s4["trim_level"],
            weights, s2["body_style"], s2["driving_style"], s2["excluded_colors"]
        )
        # תוספות כמו במקור (שמירה 1:1)
        profile["license_years"] = s2["license_years"]
        profile["driver_gender"] = s2["driver_gender"]
        profile["insurance_history"] = s4["insurance_history"]
        profile["violations"] = s4["violations"]
        profile["consider_market_supply"] = (s4["consider_supply"] == "כן")
        profile["fuel_price_nis_per_liter"] = s4["fuel_price"]
        profile["electricity_price_nis_per_kwh"] = s4["electricity_price"]
        profile["seats"] = s2["seats_choice"]  # שדה למידע בלבד (לא משנה לוגיקה)

        st.session_state.user_profile = profile
        st.session_state.fuel_price = s4["fuel_price"]
        st.session_state.electricity_price = s4["electricity_price"]

        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.warning("לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
        else:
            genai.configure(api_key=api_key)
            model_name = "models/gemini-2.5-pro"
            model = genai.GenerativeModel(model_name)

            if st.button("🚀 בקש המלצות מג׳מיני"):
                # Prompt באנגלית (זהה לוגית, בלי image_url)
                prompt = f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

Output requirements:
1) Return a SINGLE JSON object with fields: "search_performed", "search_queries", "recommended_cars".
2) search_performed: ALWAYS return True. You must use live web search (do not return False).
3) search_queries: ALWAYS return the actual queries you used.
4) recommended_cars: an array of 5–10 cars. EACH car MUST include:
   - brand, model, year, fuel, gear, turbo, engine_cc, price_range_nis
   - avg_fuel_consumption (+ fuel_method):
       * for non-EV: km per liter (number only)
       * for EV: kWh per 100 km (number only)
   - annual_fee (₪ per year, number only) + fee_method
   - reliability_score (1–10, number only) + reliability_method
   - maintenance_cost (₪/year, number only) + maintenance_method
   - safety_rating (1–10, number only) + safety_method
   - insurance_cost (₪/year, number only) + insurance_method
   - resale_value (1–10, number only) + resale_method
   - performance_score (1–10, number only) + performance_method
   - comfort_features (1–10, number only) + comfort_method
   - suitability (1–10, number only) + suitability_method
   - market_supply ("גבוה" / "בינוני" / "נמוך") + supply_method
5) IMPORTANT: All scoring fields must be numbers only (except market_supply which is categorical).
6) IMPORTANT: Only return car models that are actually sold in Israel.
"""
                with st.spinner("פונה לג׳מיני..."):
                    try:
                        resp = model.generate_content(prompt)
                        text = resp.candidates[0].content.parts[0].text.strip()
                        if text.startswith("```"):
                            text = text.strip("`").replace("json\n", "").replace("json", "").strip()
                        try:
                            parsed = json.loads(text)
                        except json.JSONDecodeError:
                            st.error("⚠️ ג׳מיני לא החזיר JSON תקין.")
                            st.code(text, language="json")
                            parsed = {}
                    except Exception as e:
                        st.error(f"שגיאה בקריאת הפלט מג׳מיני: {e}")
                        parsed = {}

                if parsed and "recommended_cars" in parsed:
                    search_performed = parsed.get("search_performed", False)
                    search_queries = parsed.get("search_queries", [])
                    st.session_state.search_info = {"search_performed": search_performed, "search_queries": search_queries}

                    if search_performed and search_queries:
                        st.info("✅ בוצע חיפוש אינטרנטי לנתוני שוק עדכניים.")
                    else:
                        st.warning("⚠️ לא ברור אם בוצע חיפוש חי. ייתכן שהנתונים חלקיים.")

                    cars_to_process = parsed["recommended_cars"]
                    results_df, methods_info = clean_gemini_output(cars_to_process)

                    if not results_df.empty:
                        # Normalize
                        results_df = normalize_car_values(results_df)

                        if "avg_fuel_consumption" not in results_df.columns:
                            st.error("חסר שדה avg_fuel_consumption בפלט.")
                            st.stop()

                        is_ev = results_df["fuel"].str.lower().eq("electric")
                        km_per_liter = results_df["avg_fuel_consumption"].where(~is_ev, np.nan).replace(0, np.nan)
                        kwh_per_100km = results_df["avg_fuel_consumption"].where(is_ev, np.nan)

                        annual_km = profile["annual_km"]
                        fuel_price = st.session_state.fuel_price or 7.0
                        elec_price = st.session_state.electricity_price or 0.65

                        fuel_cost = (annual_km / km_per_liter) * fuel_price
                        elec_cost = (annual_km / 100.0) * kwh_per_100km * elec_price

                        results_df["annual_energy_cost"] = np.where(is_ev, elec_cost, fuel_cost)
                        # לשמירת תאימות לאחור – אבל לא נציג את העמודה האנגלית בהמשך
                        results_df["annual_fuel_cost"] = results_df["annual_energy_cost"]

                        for col in ["maintenance_cost", "insurance_cost", "annual_fee"]:
                            if col not in results_df.columns:
                                results_df[col] = 0.0

                        results_df["total_annual_cost"] = (
                            results_df["annual_energy_cost"].fillna(0) +
                            results_df["maintenance_cost"].fillna(0) +
                            results_df["insurance_cost"].fillna(0) +
                            results_df["annual_fee"].fillna(0)
                        )

                        # כותרות צריכה דינמיות בעברית
                        if results_df["fuel"].str.lower().eq("electric").any():
                            column_map_he["avg_fuel_consumption"] = "צריכת חשמל (קוט\"ש/100 ק\"מ)"
                            column_map_he["annual_energy_cost"] = "עלות חשמל שנתית (₪)"
                        else:
                            column_map_he["avg_fuel_consumption"] = "צריכת דלק ממוצעת (ק\"מ/ל')"
                            column_map_he["annual_energy_cost"] = "עלות דלק שנתית (₪)"

                        results_df_display = results_df.copy()
                        # הסרת כפילות: לא מציגים את העמודה האנגלית
                        if "annual_fuel_cost" in results_df_display.columns:
                            results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])

                        results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
                        results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
                        results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
                        results_df_display = results_df_display.rename(columns=column_map_he)

                        # לוודא שגם אחרי rename אין עמודה אנגלית מיותרת
                        if "annual_fuel_cost" in results_df_display.columns:
                            results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])

                        st.session_state.results_df = results_df
                        st.session_state.methods_info = methods_info
                        st.success(f"✅ התקבלו {len(results_df)} רכבים מג׳מיני.")

                else:
                    st.error("⚠️ לא נמצאו רכבים בפלט.")

    # תוצאות (אם קיימות)
    results_df = st.session_state.results_df
    methods_info = st.session_state.methods_info or []

    if results_df is not None and not results_df.empty:
        results_df_display = results_df.copy()
        # הסרת העמודה האנגלית אם קיימת
        if "annual_fuel_cost" in results_df_display.columns:
            results_df_display = results_df_display.drop(columns=["annual_fuel_cost"])

        results_df_display["fuel"] = results_df_display["fuel"].map(fuel_map_he).fillna(results_df_display["fuel"])
        results_df_display["gear"] = results_df_display["gear"].map(gear_map_he).fillna(results_df_display["gear"])
        results_df_display["turbo"] = results_df_display["turbo"].map(turbo_map_he).fillna(results_df_display["turbo"])
        results_df_display = results_df_display.rename(columns=column_map_he)

        st.markdown('<div class="disclaimer">⚠️ הנתונים הם הערכה גסה של AI; יש לאמת לפני קנייה.</div>', unsafe_allow_html=True)
        st.dataframe(results_df_display.reset_index(drop=True), use_container_width=True)

        # גרף עלות כוללת שנתית (נשאר)
        st.markdown("### 📊 השוואת עלות כוללת שנתית")
        chart_df = results_df_display[["מותג", "דגם", "שנה", "עלות כוללת שנתית (₪)"]].copy()
        chart_df["רכב"] = chart_df["מותג"] + " " + chart_df["דגם"] + " " + chart_df["שנה"].astype(str)
        chart_df = chart_df.set_index("רכב")
        st.bar_chart(chart_df["עלות כוללת שנתית (₪)"])

        # הסברים מפורטים בעברית לכל רכב (במקום methods באנגלית)
        st.markdown("### 📝 הסברים מפורטים לכל רכב")
        for i, row in results_df_display.iterrows():
            car_name = f"{row['מותג']} {row['דגם']} {row['שנה']}"
            with st.expander(f"📝 הסבר מפורט על {car_name}"):
                st.caption(f"דלק: {row['דלק']} | תיבה: {row['תיבה']} | טורבו: {row['טורבו']}")
                if "טווח מחיר (₪)" in row:
                    st.write(f"**טווח מחיר:** {row['טווח מחיר (₪)']}")
                st.write(f"**עלות שנתית:** {float(row['עלות כוללת שנתית (₪)']):.0f} ₪")

                method = methods_info[i] if i < len(methods_info) else {}
                if method:
                    # יצירת טקסט בעברית לפי המיפוי הקיים
                    lines = []
                    for k, v in method.items():
                        field_he = method_map_he.get(k, k)
                        lines.append(f"- **{field_he}:** {v}")
                    st.markdown("\n".join(lines))
                else:
                    st.write("אין הסברים מפורטים זמינים לפריט זה.")

    st.markdown('</div>', unsafe_allow_html=True)

    def go_back(): st.session_state.ui_step = 4
    nav_buttons(left_action=go_back, show_right=False)
