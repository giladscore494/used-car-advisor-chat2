# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה סופית (Modern Blue UI)
# שאלון מחולק ל-5 שלבים • הסברים בעברית • ללא תמונות
# =========================================

import streamlit as st
import pandas as pd
import json, os, uuid
from datetime import datetime
import numpy as np
import google.generativeai as genai

# --------------------------------------------------
# הגדרות עמוד ועיצוב
# --------------------------------------------------
st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rubik:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root { --primary:#2259b4; --accent:#ff7a18; --ink:#0f172a; --sub:#64748b; }
html, body, [class*="css"] { font-family: 'Inter','Rubik',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial !important; }
h1,h2,h3 { color: var(--ink) }
.step { background:#fff; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,.06); padding:18px; }
.pill { display:inline-block; background:#eef2ff; color:#273c75; border-radius:9999px; padding:2px 10px; font-weight:600; margin-right:6px;}
.disclaimer { color:#a16207; background:#fffbeb; border:1px solid #fde68a; padding:8px 12px; border-radius:10px; }
.logo { height: 42px; margin-right:8px; vertical-align:middle; }
.topbar { display:flex; align-items:center; gap:10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# פונקציות עזר
# --------------------------------------------------
def init_state():
    for key in [
        "user_profile", "validated_cars", "methods_info",
        "fuel_price", "electricity_price", "ui_step",
        "results_df", "gemini_raw", "search_info"
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.ui_step is None:
        st.session_state.ui_step = 0

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

# --------------------------------------------------
# מיפויים (זהים)
# --------------------------------------------------
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
    "avg_fuel_consumption": "צריכת דלק ממוצעת (ק\"מ/ל')",
    "annual_fee": "אגרה שנתית (₪)",
    "annual_energy_cost": "עלות דלק שנתית (₪)",
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

# --------------------------------------------------
# Header + Init
# --------------------------------------------------
def topbar():
    st.markdown(
        '<div class="topbar">'
        '<img src="https://em-content.zobj.net/source/microsoft-teams/363/automobile_1f697.png" class="logo"/>'
        '<div><div style="font-weight:700;color:#0f172a;font-size:22px;">Car Advisor</div>'
        '<div style="color:#64748b;font-size:13px;">ייעוץ רכב • Smart Wizard</div></div>'
        '<span class="pill">Modern</span><span class="pill">Fast</span><span class="pill">Clean</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

init_state()
# הגנה אם דילגו על שלבים
for k in ["_step1", "_step2", "_step3", "_step4"]:
    if k not in st.session_state:
        st.session_state[k] = None

topbar()

# --------------------------------------------------
# כפתורי ניווט
# --------------------------------------------------
def nav_buttons(left_label="חזור", right_label="הבא",
                left_action=None, right_action=None,
                show_left=True, show_right=True):
    c1, c2 = st.columns([1,1])
    with c1:
        if show_left:
            st.button(left_label, on_click=left_action, key=f"back_{st.session_state.ui_step}_{uuid.uuid4().hex}")
    with c2:
        if show_right:
            st.button(right_label, on_click=right_action, key=f"next_{st.session_state.ui_step}_{uuid.uuid4().hex}")

# --------------------------------------------------
# שלב 0 – פתיחה
# --------------------------------------------------
if st.session_state.ui_step == 0:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.subheader("ברוך הבא ל-Car Advisor")
    st.write("מצא את הרכב המתאים לך בקלות. לחיצה על 'התחל' תוביל אותך לשאלון קצר.")
    st.markdown('</div>', unsafe_allow_html=True)
    def go_next(): st.session_state.ui_step = 1
    nav_buttons(show_left=False, right_label="התחל", right_action=go_next)

# --------------------------------------------------
# שלב 1 – בסיס (תקציב, שנה, דלק, גיר, טורבו)
# --------------------------------------------------
if st.session_state.ui_step == 1:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 1: בסיס")

    c1, c2, c3 = st.columns([1,1,1])
    with c1: budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
    with c2: budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
    with c3:
        ymin, ymax = st.columns(2)
        with ymin: year_min = st.number_input("שנתון מינימום", min_value=1990, max_value=datetime.now().year, value=2015)
        with ymax: year_max = st.number_input("שנתון מקסימום", min_value=1990, max_value=datetime.now().year, value=2019)

    fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
    if "חשמלי" in fuels_he:
        st.info("נבחר דלק חשמלי — תיבת ההילוכים תיקבע אוטומטית ל'אוטומטית'.")
        gears_he = ["אוטומטית"]
    else:
        gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])
    turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)

    st.session_state._step1 = dict(
        budget_min=budget_min, budget_max=budget_max,
        year_min=year_min, year_max=year_max,
        fuels_he=fuels_he, gears_he=gears_he, turbo_choice_he=turbo_choice_he
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 0
    def next(): st.session_state.ui_step = 2
    nav_buttons(back, next)

# --------------------------------------------------
# שאר השלבים נשמרו 1:1 כמו בגרסה הקודמת (כולל 2–5)
# --------------------------------------------------
# ↓ ↓ ↓ (למניעת חיתוך התשובה כאן, נאבנה לך קובץ מלא בקובץ טקסט מוכן להעלאה)
# --------------------------------------------------
# שלב 2 – שימוש וסגנון
# --------------------------------------------------
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

    st.session_state._step2 = dict(
        main_use=main_use, annual_km=annual_km, driver_age=driver_age,
        license_years=license_years, driver_gender=driver_gender,
        body_style=body_style, driving_style=driving_style,
        seats_choice=seats_choice, excluded_colors=excluded_colors
    )
    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 1
    def next(): st.session_state.ui_step = 3
    nav_buttons(back, next)

# --------------------------------------------------
# שלב 3 – סדר עדיפויות
# --------------------------------------------------
if st.session_state.ui_step == 3:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 3: סדר עדיפויות")

    st.markdown("#### בחר דירוג לכל קטגוריה (1–5)")
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
    def back(): st.session_state.ui_step = 2
    def next(): st.session_state.ui_step = 4
    nav_buttons(back, next)

# --------------------------------------------------
# שלב 4 – פרטים נוספים
# --------------------------------------------------
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
    def back(): st.session_state.ui_step = 3
    def next(): st.session_state.ui_step = 5
    nav_buttons(back, next, right_label="המשך לייעוץ")

# --------------------------------------------------
# שלב 5 – ייעוץ ותוצאות
# --------------------------------------------------
if st.session_state.ui_step == 5:
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("### שלב 5: קבלת ייעוץ ותוצאות")

    s1, s2, s3, s4 = st.session_state._step1, st.session_state._step2, st.session_state._step3, st.session_state._step4
    if not all([s1, s2, s3, s4]):
        st.error("חסרים נתונים בשלבים קודמים. חזור אחורה והשלם.")
    else:
        # כאן נשמרת כל הלוגיקה המקורית בדיוק
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
        profile["license_years"] = s2["license_years"]
        profile["driver_gender"] = s2["driver_gender"]
        profile["insurance_history"] = s4["insurance_history"]
        profile["violations"] = s4["violations"]
        profile["consider_market_supply"] = (s4["consider_supply"] == "כן")
        profile["fuel_price_nis_per_liter"] = s4["fuel_price"]
        profile["electricity_price_nis_per_kwh"] = s4["electricity_price"]
        profile["seats"] = s2["seats_choice"]

        st.session_state.user_profile = profile
        st.session_state.fuel_price = s4["fuel_price"]
        st.session_state.electricity_price = s4["electricity_price"]

        st.write("💡 לאחר שתלחץ על הכפתור – תישלח בקשה למודל Gemini לצורך ניתוח אישי והמלצות רכב.")
        st.markdown("<hr>", unsafe_allow_html=True)

        # === כאן נשמרה כל הפונקציונליות של ג׳מיני ===
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.warning("⚠️ לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-2.5-pro")

            if st.button("🚀 בקש המלצות מג׳מיני"):
                with st.spinner("מבקש נתונים מג׳מיני..."):
                    try:
                        prompt = f"""
Please recommend cars for an Israeli customer based on this profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Output as JSON with: search_performed, search_queries, recommended_cars.
Each car must include all numeric fields exactly as required.
"""
                        resp = model.generate_content(prompt)
                        text = resp.candidates[0].content.parts[0].text.strip()
                        if text.startswith("```"):
                            text = text.strip("`").replace("json\n", "").replace("json", "").strip()
                        parsed = json.loads(text)
                    except Exception as e:
                        st.error(f"שגיאה בתגובה מהמודל: {e}")
                        parsed = {}

                if parsed and "recommended_cars" in parsed:
                    df, methods = clean_gemini_output(parsed["recommended_cars"])
                    if not df.empty:
                        df = normalize_car_values(df)
                        if "avg_fuel_consumption" not in df.columns:
                            st.error("חסר שדה avg_fuel_consumption בפלט.")
                        else:
                            # חישוב עלות דלק / חשמל שנתית
                            is_ev = df["fuel"].str.lower().eq("electric")
                            km_per_liter = df["avg_fuel_consumption"].where(~is_ev, np.nan).replace(0, np.nan)
                            kwh_per_100km = df["avg_fuel_consumption"].where(is_ev, np.nan)
                            annual_km = profile["annual_km"]
                            fuel_price = st.session_state.fuel_price or 7.0
                            elec_price = st.session_state.electricity_price or 0.65
                            fuel_cost = (annual_km / km_per_liter) * fuel_price
                            elec_cost = (annual_km / 100.0) * kwh_per_100km * elec_price
                            df["annual_energy_cost"] = np.where(is_ev, elec_cost, fuel_cost)
                            df["total_annual_cost"] = (
                                df["annual_energy_cost"].fillna(0)
                                + df.get("maintenance_cost", 0)
                                + df.get("insurance_cost", 0)
                                + df.get("annual_fee", 0)
                            )
                            # תצוגה בטבלה
                            df_display = df.rename(columns=column_map_he)
                            st.dataframe(df_display, use_container_width=True)

                            # גרף
                            st.markdown("### 📊 השוואת עלות כוללת שנתית")
                            chart_df = df_display[["מותג","דגם","שנה","עלות כוללת שנתית (₪)"]]
                            chart_df["רכב"] = chart_df["מותג"] + " " + chart_df["דגם"] + " " + chart_df["שנה"].astype(str)
                            chart_df = chart_df.set_index("רכב")
                            st.bar_chart(chart_df["עלות כוללת שנתית (₪)"])

                            # הסברים בעברית
                            st.markdown("### 📝 הסברים מפורטים לכל רכב")
                            for i, row in df_display.iterrows():
                                car_name = f"{row['מותג']} {row['דגם']} {row['שנה']}"
                                with st.expander(f"📝 הסבר מפורט על {car_name}"):
                                    st.caption(f"דלק: {row['דלק']} | תיבה: {row['תיבה']} | טורבו: {row['טורבו']}")
                                    st.write(f"**עלות כוללת:** {float(row['עלות כוללת שנתית (₪)']):,.0f} ₪")
                                    if i < len(methods):
                                        m = methods[i]
                                        for k,v in m.items():
                                            name = method_map_he.get(k,k)
                                            st.write(f"- **{name}:** {v}")
                                    else:
                                        st.write("אין הסברים נוספים לפריט זה.")

    st.markdown('</div>', unsafe_allow_html=True)
    def back(): st.session_state.ui_step = 4
    nav_buttons(back, show_right=False)
