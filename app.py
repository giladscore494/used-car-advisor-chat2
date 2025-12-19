# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – Modern Blue (Stable)
# 5 Steps • Hebrew UI • Gemini + Google Search (Grounding)
# + Tool/Grounding validation extracted from Response
# + Hardening: app-level try/except showing full traceback in UI
# =========================================

import os, json, uuid, traceback
from datetime import datetime

import streamlit as st
import pandas as pd

from google import genai
from google.genai import types as genai_types


# ---------------------------
# Streamlit config + CSS
# ---------------------------
st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rubik:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root { --primary:#2259b4; --accent:#ff7a18; --ink:#0f172a; --sub:#64748b; }
html, body, [class*="css"] { font-family: 'Inter','Rubik',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial !important; }
h1,h2,h3 { color: var(--ink) }
.step { background:#fff; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,.06); padding:18px; margin-bottom:20px; }
.pill { display:inline-block; background:#eef2ff; color:#273c75; border-radius:9999px; padding:2px 10px; font-weight:600; margin-right:6px;}
.disclaimer { color:#a16207; background:#fffbeb; border:1px solid #fde68a; padding:8px 12px; border-radius:10px; }
.logo { height: 42px; margin-right:8px; vertical-align:middle; }
.topbar { display:flex; align-items:center; gap:10px; }
.small { color:#64748b; font-size:12px; }
.badge-ok { display:inline-block; padding:2px 10px; border-radius:9999px; background:#dcfce7; color:#166534; font-weight:700; }
.badge-warn { display:inline-block; padding:2px 10px; border-radius:9999px; background:#fef9c3; color:#854d0e; font-weight:700; }
.badge-bad { display:inline-block; padding:2px 10px; border-radius:9999px; background:#fee2e2; color:#991b1b; font-weight:700; }
.codebox { background:#0b1020; color:#e5e7eb; border-radius:12px; padding:12px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:12px; overflow:auto; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 16px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Compatibility: rerun
# ---------------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ---------------------------
# Gemini config
# ---------------------------
GEMINI_MODEL_ID = "gemini-3-flash-preview"

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "⚠️ לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה."
    try:
        return genai.Client(api_key=api_key), None
    except Exception as e:
        return None, f"שגיאה באתחול לקוח Gemini: {e}"

gemini_client, gemini_init_error = get_gemini_client()


# ---------------------------
# Data helpers
# ---------------------------
fuel_map = {"בנזין": "gasoline", "היברידי": "hybrid", "דיזל היברידי": "hybrid-diesel", "דיזל": "diesel", "חשמלי": "electric"}
gear_map = {"אוטומטית": "automatic", "ידנית": "manual"}
turbo_map = {"לא משנה": "any", "כן": "yes", "לא": "no"}

column_map_he = {
    "brand":"מותג","model":"דגם","year":"שנה","fuel":"דלק","gear":"תיבה","turbo":"טורבו","engine_cc":"נפח מנוע (סמ\"ק)",
    "price_range_nis":"טווח מחיר (₪)","avg_fuel_consumption":"צריכת דלק/אנרגיה",
    "annual_fee":"אגרה שנתית (₪)","reliability_score":"אמינות","maintenance_cost":"עלות אחזקה (₪/שנה)",
    "safety_rating":"בטיחות","insurance_cost":"עלות ביטוח (₪/שנה)","resale_value":"שמירת ערך",
    "performance_score":"ביצועים","comfort_features":"נוחות","suitability":"התאמה","market_supply":"היצע בשוק",
    "fit_score":"ציון התאמה (0–100)","comparison_comment":"הערה השוואתית","not_recommended_reason":"למה לא מומלץ",
}

method_map_he = {
    "fuel_method":"שיטת חישוב צריכת דלק/חשמל","fee_method":"שיטת חישוב אגרה","reliability_method":"שיטת חישוב אמינות",
    "maintenance_method":"שיטת חישוב עלות אחזקה","safety_method":"שיטת חישוב בטיחות","insurance_method":"שיטת חישוב ביטוח",
    "resale_method":"שיטת חישוב שמירת ערך","performance_method":"שיטת חישוב ביצועים","comfort_method":"שיטת חישוב נוחות",
    "suitability_method":"שיטת חישוב התאמה","supply_method":"שיטת קביעת היצע",
}

def make_user_profile(
    budget_min, budget_max, year_min, year_max, fuels, gears, turbo_required,
    main_use, annual_km, driver_age, family_size, cargo_need, safety_required,
    trim_level, weights, body_style, driving_style, excluded_colors,
    license_years, driver_gender, insurance_history, violations,
    consider_supply, fuel_price, electricity_price, seats_choice,
):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(year_min), int(year_max)],
        "fuel": [str(f).lower() for f in fuels],
        "gear": [str(g).lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": (main_use or "").strip(),
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
        "license_years": int(license_years),
        "driver_gender": driver_gender,
        "insurance_history": insurance_history,
        "violations": violations,
        "consider_market_supply": (consider_supply == "כן"),
        "fuel_price_nis_per_liter": float(fuel_price),
        "electricity_price_nis_per_kwh": float(electricity_price),
        "seats": seats_choice,
    }

def clean_gemini_output(cars_raw):
    records, methods = [], []
    for car in cars_raw:
        if not isinstance(car, dict):
            continue
        rec, met = {}, {}
        for k, v in car.items():
            if str(k).endswith("_method"):
                met[k] = v
            else:
                rec[k] = v
        records.append(rec)
        methods.append(met)
    return pd.DataFrame(records), methods

def normalize_car_values(df):
    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].replace({
            "בנזין": "gasoline",
            "דיזל": "diesel",
            "היברידי": "hybrid",
            "דיזל היברידי": "hybrid-diesel",
            "חשמלי": "electric",
        })
    if "gear" in df.columns:
        df["gear"] = df["gear"].replace({
            "אוטומטי": "automatic",
            "אוטומטית": "automatic",
            "ידני": "manual",
            "ידנית": "manual",
        })
    if "turbo" in df.columns:
        df["turbo"] = df["turbo"].replace({"כן": True, "לא": False, True: True, False: False})
    return df


# ---------------------------
# Grounding extraction (best-effort)
# ---------------------------
def _safe_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    for m in ("model_dump", "dict", "to_dict"):
        if hasattr(obj, m):
            try:
                return getattr(obj, m)()
            except Exception:
                pass
    try:
        return dict(obj.__dict__)
    except Exception:
        return None

def extract_grounding_info(resp) -> dict:
    info = {"has_grounding_metadata": False, "sources": [], "tool_signals": []}
    if resp is None:
        return info

    cand = None
    try:
        cands = getattr(resp, "candidates", None)
        if cands:
            cand = cands[0]
    except Exception:
        cand = None

    gm = None
    try:
        gm = getattr(cand, "grounding_metadata", None) if cand is not None else getattr(resp, "grounding_metadata", None)
    except Exception:
        gm = None

    if gm is not None:
        info["has_grounding_metadata"] = True
        gm_dict = _safe_to_dict(gm) or {}

        chunks = gm_dict.get("grounding_chunks") or gm_dict.get("groundingChunks") or gm_dict.get("chunks") or []
        for ch in chunks[:20]:
            chd = _safe_to_dict(ch) or {}
            web = chd.get("web") or chd.get("retrieved_context") or chd.get("retrievedContext") or {}
            webd = _safe_to_dict(web) or {}
            uri = webd.get("uri") or webd.get("url") or chd.get("uri") or chd.get("url")
            title = webd.get("title") or chd.get("title")
            if uri or title:
                info["sources"].append({"title": title or "", "uri": uri or ""})

        supports = gm_dict.get("grounding_supports") or gm_dict.get("groundingSupports") or gm_dict.get("supports") or []
        if supports:
            info["tool_signals"].append(f"grounding_supports={len(supports)}")

    # parts-based signals
    try:
        if cand is not None:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                for p in parts:
                    pdict = _safe_to_dict(p) or {}
                    if "function_call" in pdict or "functionCall" in pdict:
                        info["tool_signals"].append("function_call_detected")
                    if "tool" in pdict or "tool_code" in pdict or "toolCode" in pdict:
                        info["tool_signals"].append("tool_part_detected")
    except Exception:
        pass

    info["tool_signals"] = sorted(set(info["tool_signals"]))
    return info


# ---------------------------
# Gemini call
# ---------------------------
def call_gemini_with_search(profile: dict) -> dict:
    if gemini_client is None:
        return {"data": {"_error": gemini_init_error or "Gemini client unavailable."}, "grounding": {}, "raw_text": ""}

    prompt = f"""
Please recommend cars for an Israeli customer. Here is the user profile (JSON):
{json.dumps(profile, ensure_ascii=False, indent=2)}

CRITICAL: Use the Google Search tool to verify Israel availability, realistic used prices in NIS, consumption values, and common issues.

Return ONE JSON object with fields:
- search_performed (true)
- search_queries (array of max 6 Hebrew queries)
- recommended_cars (array of 5-10 cars)

Each car must include the fields described previously + *_method fields in Hebrew.
Return ONLY raw JSON.
"""

    search_tool = genai_types.Tool(google_search=genai_types.GoogleSearch())
    config = genai_types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        tools=[search_tool],
        response_mime_type="application/json",
    )

    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=config,
        )
        raw_text = (getattr(resp, "text", "") or "").strip()
        grounding = extract_grounding_info(resp)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            data = {"_error": "JSON decode error from Gemini", "_raw": raw_text}

        return {"data": data, "grounding": grounding, "raw_text": raw_text}
    except Exception as e:
        return {"data": {"_error": f"Gemini call failed: {e}"}, "grounding": {}, "raw_text": ""}


# ---------------------------
# Header + navigation
# ---------------------------
def topbar():
    st.markdown(
        '<div class="topbar">'
        '<img src="https://em-content.zobj.net/source/microsoft-teams/363/automobile_1f697.png" class="logo"/>'
        '<div>'
        '<div style="font-weight:700;color:#0f172a;font-size:22px;">Car Advisor</div>'
        '<div style="color:#64748b;font-size:13px;">ייעוץ רכב • Smart Wizard</div>'
        '</div>'
        '<span class="pill">Modern</span><span class="pill">Fast</span><span class="pill">Grounded</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

def nav_buttons(left_label="חזור", right_label="הבא", left_action=None, right_action=None, show_left=True, show_right=True):
    c1, c2 = st.columns([1, 1])
    with c1:
        if show_left:
            st.button(left_label, on_click=left_action, key=f"back_{st.session_state.ui_step}_{uuid.uuid4().hex}")
    with c2:
        if show_right:
            st.button(right_label, on_click=right_action, key=f"next_{st.session_state.ui_step}_{uuid.uuid4().hex}")


# ---------------------------
# State init
# ---------------------------
if "ui_step" not in st.session_state:
    st.session_state.ui_step = 0
for k in ["_step1","_step2","_step3","_step4","results_payload","search_info","raw_text","user_profile"]:
    if k not in st.session_state:
        st.session_state[k] = None

topbar()


# ==================================================
# MAIN APP (wrapped to show real errors in UI)
# ==================================================
try:
    # -------------------------
    # Step 0
    # -------------------------
    if st.session_state.ui_step == 0:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.subheader("ברוך הבא ל-Car Advisor")
        st.write("לחץ 'התחל' כדי לעבור לשאלון.")
        if gemini_client is None:
            st.markdown(f'<div class="disclaimer">{gemini_init_error}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        def go_next():
            st.session_state.ui_step = 1

        nav_buttons(show_left=False, right_label="התחל", right_action=go_next)

    # -------------------------
    # Step 1
    # -------------------------
    elif st.session_state.ui_step == 1:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.markdown("### שלב 1: פרטים בסיסיים")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
        with c2:
            budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
        with c3:
            ymin, ymax = st.columns(2)
            with ymin:
                year_min = st.number_input("שנתון מינימום", 1990, datetime.now().year, 2015)
            with ymax:
                year_max = st.number_input("שנתון מקסימום", 1990, datetime.now().year, 2019)

        fuels_he = st.multiselect("סוגי דלק מועדפים", list(fuel_map.keys()), default=["בנזין"])
        if "חשמלי" in fuels_he:
            st.info("נבחר דלק חשמלי — תיבת ההילוכים תוגדר כ'אוטומטית'.")
            gears_he = ["אוטומטית"]
        else:
            gears_he = st.multiselect("תיבת הילוכים", list(gear_map.keys()), default=["אוטומטית"])

        turbo_choice_he = st.selectbox("טורבו?", list(turbo_map.keys()), index=1)

        st.session_state._step1 = {
            "budget_min": budget_min,
            "budget_max": budget_max,
            "year_min": year_min,
            "year_max": year_max,
            "fuels_he": fuels_he,
            "gears_he": gears_he,
            "turbo_choice_he": turbo_choice_he,
        }

        st.markdown("</div>", unsafe_allow_html=True)

        nav_buttons(
            left_label="חזור",
            right_label="הבא",
            left_action=lambda: setattr(st.session_state, "ui_step", 0),
            right_action=lambda: setattr(st.session_state, "ui_step", 2),
        )

    # -------------------------
    # Step 2
    # -------------------------
    elif st.session_state.ui_step == 2:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.markdown("### שלב 2: שימוש וסגנון")

        c4, c5, c6 = st.columns([2, 1, 1])
        with c4:
            main_use = st.text_area("תיאור הרכב והשימוש בו", value="נסיעה יומיומית לעבודה וטיולים קצרים", height=100)
        with c5:
            annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
        with c6:
            driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=21)

        c6a, c6b = st.columns(2)
        with c6a:
            license_years = st.number_input("וותק רישיון (שנים)", min_value=0, max_value=50, value=2)
        with c6b:
            driver_gender = st.selectbox("מין נהג", ["זכר", "נקבה"])

        cstyle1, cstyle2, cseats = st.columns([1, 1, 1])
        with cstyle1:
            body_style = st.selectbox("סגנון מרכב מועדף", ["כללי", "סדאן", "האצ'בק", "קרוסאובר/ג'יפון"])
        with cstyle2:
            driving_style = st.selectbox("סגנון נהיגה", ["רגוע ונינוח", "דינמי וספורטיבי"])
        with cseats:
            seats_choice = st.selectbox("מספר מקומות", ["4", "5", "5+"])

        excluded_colors_raw = st.text_input("צבעים לפסילה (מופרדים בפסיק)", value="")
        excluded_colors = [c.strip() for c in excluded_colors_raw.split(",") if c.strip()]

        st.session_state._step2 = {
            "main_use": main_use,
            "annual_km": annual_km,
            "driver_age": driver_age,
            "license_years": license_years,
            "driver_gender": driver_gender,
            "body_style": body_style,
            "driving_style": driving_style,
            "seats_choice": seats_choice,
            "excluded_colors": excluded_colors,
        }

        st.markdown("</div>", unsafe_allow_html=True)

        nav_buttons(
            left_label="חזור",
            right_label="הבא",
            left_action=lambda: setattr(st.session_state, "ui_step", 1),
            right_action=lambda: setattr(st.session_state, "ui_step", 3),
        )

    # -------------------------
    # Step 3
    # -------------------------
    elif st.session_state.ui_step == 3:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.markdown("### שלב 3: סדר עדיפויות")
        st.markdown("#### בחר דירוג לכל קטגוריה (1–5)")

        reliability_weight = st.slider("אמינות", 1, 5, 5)
        resale_weight = st.slider("שמירת ערך", 1, 5, 3)
        fuel_weight = st.slider("חיסכון בדלק", 1, 5, 4)
        performance_weight = st.slider("ביצועים", 1, 5, 2)
        comfort_weight = st.slider("נוחות", 1, 5, 3)

        st.session_state._step3 = {
            "weights": {
                "reliability": reliability_weight,
                "resale": resale_weight,
                "fuel": fuel_weight,
                "performance": performance_weight,
                "comfort": comfort_weight,
            }
        }

        st.markdown("</div>", unsafe_allow_html=True)

        nav_buttons(
            left_label="חזור",
            right_label="הבא",
            left_action=lambda: setattr(st.session_state, "ui_step", 2),
            right_action=lambda: setattr(st.session_state, "ui_step", 4),
        )

    # -------------------------
    # Step 4
    # -------------------------
    elif st.session_state.ui_step == 4:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.markdown("### שלב 4: פרטים נוספים")

        insurance_history = st.text_input("עבר ביטוחי", value="שנתיים ללא תביעות")
        violations = st.selectbox("דוחות/שלילות", ["אין", "שלילה בעבר", "נקודות פעילות"])

        cfam, ccargo, csafety, ctrim = st.columns([1, 1, 1, 1])
        with cfam:
            family_size = st.selectbox("גודל משפחה", ["1-2", "3-4", "5+"])
        with ccargo:
            cargo_need = st.selectbox("צורך בתא מטען", ["קטן", "בינוני", "גדול"])
        with csafety:
            safety_required = st.radio("חובה מערכות בטיחות אקטיביות?", ["כן", "לא"])
        with ctrim:
            trim_level = st.selectbox("רמת אבזור", ["בסיסי", "סטנדרטי", "עשיר"])

        consider_supply = st.radio("האם להתחשב בהיצע בשוק?", ["כן", "לא"], index=0)

        cfp, cep = st.columns([1, 1])
        with cfp:
            fuel_price = st.number_input("מחיר ליטר דלק (₪)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
        with cep:
            electricity_price = st.number_input("מחיר חשמל לקוט״ש (₪)", min_value=0.1, max_value=5.0, value=0.65, step=0.01)

        st.session_state._step4 = {
            "insurance_history": insurance_history,
            "violations": violations,
            "family_size": family_size,
            "cargo_need": cargo_need,
            "safety_required": safety_required,
            "trim_level": trim_level,
            "consider_supply": consider_supply,
            "fuel_price": fuel_price,
            "electricity_price": electricity_price,
        }

        st.markdown("</div>", unsafe_allow_html=True)

        nav_buttons(
            left_label="חזור",
            right_label="המשך לייעוץ",
            left_action=lambda: setattr(st.session_state, "ui_step", 3),
            right_action=lambda: setattr(st.session_state, "ui_step", 5),
        )

    # -------------------------
    # Step 5
    # -------------------------
    elif st.session_state.ui_step == 5:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        st.markdown("### שלב 5: קבלת ייעוץ ותוצאות")

        s1, s2, s3, s4 = st.session_state._step1, st.session_state._step2, st.session_state._step3, st.session_state._step4
        if not all([s1, s2, s3, s4]):
            st.error("חסרים נתונים בשלבים קודמים. חזור אחורה והשלם.")
        else:
            fuels = [fuel_map[f] for f in (s1.get("fuels_he") or []) if f in fuel_map]
            gears = [gear_map[g] for g in (s1.get("gears_he") or []) if g in gear_map]
            turbo_choice = turbo_map.get(s1.get("turbo_choice_he", "לא משנה"), "any")
            weights = (s3.get("weights") or {})

            profile = make_user_profile(
                s1["budget_min"], s1["budget_max"],
                s1["year_min"], s1["year_max"],
                fuels, gears, turbo_choice,
                s2["main_use"], s2["annual_km"], s2["driver_age"],
                s4["family_size"], s4["cargo_need"], s4["safety_required"],
                s4["trim_level"], weights,
                s2["body_style"], s2["driving_style"], s2["excluded_colors"],
                s2["license_years"], s2["driver_gender"],
                s4["insurance_history"], s4["violations"],
                s4["consider_supply"], s4["fuel_price"], s4["electricity_price"],
                s2["seats_choice"],
            )

            st.session_state.user_profile = profile

            st.markdown("#### פרופיל שנשלח למודל")
            st.markdown(f'<div class="codebox">{json.dumps(profile, ensure_ascii=False, indent=2)}</div>', unsafe_allow_html=True)

            cA, cB = st.columns([1, 1])
            with cA:
                run_btn = st.button("🚀 הרץ ייעוץ (Gemini + Search)", use_container_width=True)
            with cB:
                back_btn = st.button("⬅️ חזור לשאלון", use_container_width=True)

            if back_btn:
                st.session_state.ui_step = 4
                safe_rerun()

            if run_btn:
                if gemini_client is None:
                    st.error(gemini_init_error or "Gemini client unavailable.")
                else:
                    with st.spinner("מריץ…"):
                        result = call_gemini_with_search(profile)

                    payload = result.get("data") or {}
                    grounding = result.get("grounding") or {}
                    raw_text = result.get("raw_text") or ""

                    st.session_state.results_payload = payload
                    st.session_state.search_info = grounding
                    st.session_state.raw_text = raw_text

                    st.markdown("---")
                    st.subheader("ולידציה: האם הטול באמת הופעל (מתוך Response)?")

                    has_any = bool(grounding.get("has_grounding_metadata")) or bool(grounding.get("sources")) or bool(grounding.get("tool_signals"))
                    if has_any:
                        st.markdown('<span class="badge-ok">✅ נמצא סימן לשימוש בטול/grounding</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-warn">⚠️ אין סימן ברור ב-Response</span>', unsafe_allow_html=True)

                    st.write("has_grounding_metadata:", grounding.get("has_grounding_metadata", False))
                    st.write("tool_signals:", grounding.get("tool_signals") or "—")

                    if isinstance(payload, dict):
                        st.write("search_performed (מה-JSON):", payload.get("search_performed", "—"))
                        st.write("search_queries (מה-JSON):", payload.get("search_queries", "—"))

                    sources = grounding.get("sources") or []
                    if sources:
                        st.markdown("**מקורות (אם הוחזרו):**")
                        for i, s in enumerate(sources[:10], start=1):
                            st.write(f"{i}. {s.get('title','').strip()} — {s.get('uri','').strip()}")

                    st.markdown("---")
                    st.subheader("תוצאות")

                    if isinstance(payload, dict) and payload.get("_error"):
                        st.error(payload.get("_error"))
                        with st.expander("DEBUG: resp.text גולמי"):
                            st.markdown(f'<div class="codebox">{raw_text}</div>', unsafe_allow_html=True)
                    else:
                        cars = payload.get("recommended_cars") if isinstance(payload, dict) else None
                        if not isinstance(cars, list) or not cars:
                            st.warning("לא התקבל recommended_cars תקין.")
                            with st.expander("DEBUG: JSON גולמי"):
                                st.markdown(f'<div class="codebox">{json.dumps(payload, ensure_ascii=False, indent=2)}</div>', unsafe_allow_html=True)
                        else:
                            df, methods_list = clean_gemini_output(cars)
                            df = normalize_car_values(df)

                            preferred_cols = [
                                "brand","model","year","fuel","gear","turbo",
                                "price_range_nis","reliability_score","resale_value",
                                "avg_fuel_consumption","market_supply","fit_score",
                            ]
                            show_cols = [c for c in preferred_cols if c in df.columns] or list(df.columns)

                            df_show = df[show_cols].copy()
                            df_show.columns = [column_map_he.get(c, c) for c in df_show.columns]
                            st.dataframe(df_show, use_container_width=True, hide_index=True)

                            st.markdown("### פירוט רכבים")
                            for idx, row in df.iterrows():
                                title = f"{row.get('brand','')} {row.get('model','')} {row.get('year','')}"
                                with st.expander(title, expanded=False):
                                    for col in df.columns:
                                        if str(col).endswith("_method"):
                                            continue
                                        st.write(f"**{column_map_he.get(col, col)}:**", row.get(col, None))

                                    st.markdown("#### שיטות חישוב (methods)")
                                    m = methods_list[idx] if isinstance(methods_list, list) and idx < len(methods_list) else {}
                                    if not m:
                                        st.caption("אין *_method.")
                                    else:
                                        for mk, mv in m.items():
                                            st.write(f"**{method_map_he.get(mk, mk)}:**", mv)

                            with st.expander("DEBUG: JSON גולמי"):
                                st.markdown(f'<div class="codebox">{json.dumps(payload, ensure_ascii=False, indent=2)}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

except Exception:
    st.error("האפליקציה נתקלה בשגיאה פנימית. הנה ה-Traceback המלא (לא מרודד):")
    st.code(traceback.format_exc(), language="text")

st.markdown("---")
st.caption("אם grounding_metadata לא מגיע — זה לרוב עניין של מודל/SDK/הרשאות. עדיין ייתכן שהיה חיפוש, אבל בלי metadata מפורט.")
