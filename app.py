# app.py
# -*- coding: utf-8 -*-
# =========================================
# Car Advisor – גרסה מאוחדת עם Gemini 2.5 Pro (תיקון תגיות JSON)
# =========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json, io, os
from datetime import datetime
import google.generativeai as genai

st.set_page_config(page_title="Car Advisor", page_icon="🚗", layout="wide")

# -------- Helpers --------
def init_state():
    for key in ["inventory_df","user_profile","validated_cars","df_ranked"]:
        if key not in st.session_state:
            st.session_state[key] = None

def make_user_profile(budget_min, budget_max, years_range, fuels, gears,
                      turbo_required, main_use, annual_km, driver_age):
    return {
        "budget_nis": [float(budget_min), float(budget_max)],
        "years": [int(years_range[0]), int(years_range[1])],
        "fuel": [f.lower() for f in fuels],
        "gear": [g.lower() for g in gears],
        "turbo_required": None if turbo_required == "any" else (turbo_required == "yes"),
        "main_use": main_use.strip(),
        "annual_km": int(annual_km),
        "driver_age": int(driver_age),
    }

# -------- שלב 1: שאלון + מאגר --------
init_state()
st.title("🚗 Car Advisor – ייעוץ רכב")

st.markdown("### שלב 1: שאלון")
col1, col2, col3 = st.columns([1,1,1])
with col1: budget_min = st.number_input("תקציב מינימום (₪)", min_value=0, step=1000, value=40000)
with col2: budget_max = st.number_input("תקציב מקסימום (₪)", min_value=0, step=1000, value=65000)
with col3:
    ymin, ymax = st.columns(2)
    with ymin: year_min = st.number_input("שנתון מינימום", min_value=1990, max_value=datetime.now().year, value=2015)
    with ymax: year_max = st.number_input("שנתון מקסימום", min_value=1990, max_value=datetime.now().year, value=2019)

fuels = st.multiselect("סוגי דלק מועדפים", ["gasoline","hybrid","diesel","electric"], default=["gasoline","hybrid"])
gears = st.multiselect("תיבת הילוכים", ["automatic","manual"], default=["automatic"])
turbo_choice = st.selectbox("טורבו?", ["any","yes","no"], index=0)

c4, c5, c6 = st.columns([2,1,1])
with c4: main_use = st.text_input("שימוש עיקרי", value="city + intercity, family of 4")
with c5: annual_km = st.number_input("נסועה שנתית (ק״מ)", min_value=0, step=1000, value=15000)
with c6: driver_age = st.number_input("גיל נהג", min_value=16, max_value=100, value=35)

profile = make_user_profile(budget_min, budget_max, [year_min, year_max],
                            fuels, gears, turbo_choice, main_use, annual_km, driver_age)
st.session_state.user_profile = profile
st.json(profile)

st.markdown("### שלב 1ב: טעינת מאגר")
uploaded = st.file_uploader("בחר קובץ CSV עם עמודות: brand, model, year, engine_cc, automatic, fuel", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, encoding="utf-8-sig")
    fuel_map = {"בנזין":"gasoline","דיזל":"diesel","היברידי":"hybrid","חשמלי":"electric"}
    if df["fuel"].dtype == "object":
        df["fuel"] = df["fuel"].map(fuel_map).fillna(df["fuel"])
    if df["automatic"].dtype in ["int64","float64"]:
        df["automatic"] = df["automatic"].apply(lambda x: "automatic" if x==1 else "manual")
    st.session_state.inventory_df = df
    st.success(f"מאגר נטען ({len(df)} שורות, {df['brand'].nunique()} מותגים).")

# -------- שלב 2: Gemini + סינון --------
st.markdown("### שלב 2: Gemini – המלצות ראשוניות")
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("לא נמצא GEMINI_API_KEY בסודות או במשתני סביבה.")
else:
    genai.configure(api_key=api_key)
    model_name = "models/gemini-2.5-pro"
    model = genai.GenerativeModel(model_name)

    if st.button("🚀 בקש המלצות מגימניי"):
        prompt = f"""
        אני צריך המלצות לרכבים. אלה התכונות שהלקוח חיפש:
        {json.dumps(profile, ensure_ascii=False, indent=2)}

        שלבים:
        1. חשוב לפי הנתונים בשאלון.
        2. בצע חיפוש ברשת למחירים עדכניים ולזמינות הדגמים בישראל.
        3. סנן רק רכבים בתקציב.
        4. דרג לפי חיסכון, אמינות, עלויות תחזוקה.
        5. החזר 5–10 רכבים בלבד.
        6. החזר אך ורק בפורמט JSON תקין, בלי טקסט נוסף, לדוגמה:
        [
          {{
            "brand": "Toyota",
            "model": "Corolla",
            "year": 2018,
            "fuel": "gasoline",
            "gear": "automatic",
            "turbo": false,
            "price_range_nis": [55000, 65000],
            "notes": "אמינה, זולה לתחזוקה"
          }}
        ]
        """
        try:
            resp = model.generate_content(prompt)
            text = resp.candidates[0].content.parts[0].text.strip()

            # ניקוי תגיות ```json או ```
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json\n", "").replace("json", "").strip()

            # נסיון לפענח JSON
            try:
                cars_from_gemini = json.loads(text)
            except json.JSONDecodeError:
                st.error("⚠️ הפלט מגימניי לא היה JSON טהור. להלן מה שהתקבל:")
                st.code(text)
                cars_from_gemini = []

        except Exception as e:
            st.error(f"שגיאה בקריאת הפלט מגימניי: {e}")
            cars_from_gemini = []

        valid_cars = []
        if st.session_state.inventory_df is not None:
            df_inv = st.session_state.inventory_df
            min_budget, max_budget = profile["budget_nis"]
            min_budget, max_budget = min_budget*0.91, max_budget*1.09
            for car in cars_from_gemini:
                in_inv = not df_inv[
                    (df_inv["brand"].str.lower() == str(car["brand"]).lower()) &
                    (df_inv["model"].str.lower() == str(car["model"]).lower()) &
                    (df_inv["year"] == car["year"])
                ].empty
                price_ok = min_budget <= car["price_range_nis"][0] and max_budget >= car["price_range_nis"][1]
                if in_inv and price_ok:
                    valid_cars.append(car)

        st.session_state.validated_cars = pd.DataFrame(valid_cars)
        if not st.session_state.validated_cars.empty:
            st.success(f"נמצאו {len(st.session_state.validated_cars)} רכבים אחרי סינון.")

# -------- שלב 3: FitScore --------
def calculate_fit_score(row, profile):
    score = 0; max_score = 100
    price_min, price_max = profile["budget_nis"]
    budget_mid = (price_min + price_max) / 2
    car_mid = (row["price_range_nis"][0] + row["price_range_nis"][1]) / 2
    price_diff = abs(car_mid - budget_mid) / (budget_mid if budget_mid>0 else 1)
    price_score = max(0, 30 - price_diff*30)
    if car_mid <= budget_mid: price_score += 5
    score += price_score

    year_min, year_max = profile["years"]
    year_range = max(1, year_max-year_min)
    year_score = ((row["year"]-year_min)/year_range)*20
    score += max(0, min(20, year_score))

    if str(row["fuel"]).lower() in profile["fuel"]: score += 10
    if str(row["gear"]).lower() in profile["gear"]: score += 10

    if profile["turbo_required"] is not None:
        if profile["turbo_required"] == row.get("turbo"): score += 5

    if profile["annual_km"] > 20000 and str(row["fuel"]).lower()=="diesel":
        score += 5
    if profile["annual_km"] < 10000 and str(row["fuel"]).lower()=="electric":
        score += 5

    notes = str(row.get("notes","")).lower()
    if "אמינה" in notes or "reliable" in notes: score += 10
    if "תחזוקה" in notes and ("גבוה" in notes or "יקר" in notes or "high" in notes): score -= 5

    return round(min(score,max_score),1)

if st.session_state.validated_cars is not None and not st.session_state.validated_cars.empty:
    df = st.session_state.validated_cars.copy()
    df["FitScore"] = df.apply(lambda r: calculate_fit_score(r, profile), axis=1)
    df = df.sort_values("FitScore", ascending=False).reset_index(drop=True)
    st.session_state.df_ranked = df

    st.markdown("### שלב 3: רכבים עם ציון FitScore")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ הורד CSV עם ציונים", csv, "ranked_cars.csv", "text/csv")

# -------- שלב 4: גרפים --------
if st.session_state.df_ranked is not None and not st.session_state.df_ranked.empty:
    df = st.session_state.df_ranked
    st.markdown("### שלב 4: גרפים")

    fig, ax = plt.subplots(); df["FitScore"].plot(kind="hist", bins=10, edgecolor="black", ax=ax)
    ax.set_title("התפלגות FitScore"); st.pyplot(fig)

    fig, ax = plt.subplots()
    df.groupby("brand")["FitScore"].mean().sort_values(ascending=False).head(10).plot(kind="bar", ax=ax)
    ax.set_title("Top 10 מותגים"); st.pyplot(fig)

    df["avg_price"] = df["price_range_nis"].apply(lambda x: (x[0]+x[1])/2)
    fig, ax = plt.subplots()
    df.plot(kind="scatter", x="year", y="avg_price", c="FitScore", cmap="viridis", ax=ax, s=80)
    ax.set_title("מחיר מול שנתון"); st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(df["avg_price"], df["FitScore"], c=df["FitScore"], cmap="plasma", s=80)
    ax.set_xlabel("מחיר ממוצע"); ax.set_ylabel("FitScore")
    ax.set_title("מחיר מול FitScore – למצוא דילים"); st.pyplot(fig)

# -------- שלב 5: ייצוא --------
if st.session_state.df_ranked is not None and not st.session_state.df_ranked.empty:
    st.markdown("### שלב 5: ייצוא ושמירה")
    df = st.session_state.df_ranked

    json_data = df.to_json(orient="records", force_ascii=False, indent=2)
    st.download_button("📥 הורד JSON", json_data,
        f"cars_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Cars", index=False)
    st.download_button("📥 הורד Excel", buffer.getvalue(),
        f"cars_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
