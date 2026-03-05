"""
app.py — LoanIQ Credit Risk Predictor
======================================
Professional Streamlit frontend for Loan Approval Prediction.
German Credit Dataset · Logistic Regression · Decision Tree · Random Forest

Author  : Adil Khan
Run     : python -m streamlit run app.py
Requires: models/ folder with .pkl files (run loan_prediction.ipynb first)
"""

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ── Page config (must be first) ───────────────────────
st.set_page_config(
    page_title="LoanIQ · Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg:         #F4F6FB;
  --surface:    #FFFFFF;
  --surface2:   #EEF1F8;
  --border:     #DDE2EF;
  --navy:       #0B1F3A;
  --navy2:      #162D52;
  --blue:       #2563EB;
  --blue-lt:    #EEF3FF;
  --blue-bd:    #C3D2FD;
  --green:      #059669;
  --green-lt:   #ECFDF5;
  --green-bd:   #A7F3D0;
  --red:        #DC2626;
  --red-lt:     #FEF2F2;
  --red-bd:     #FECACA;
  --amber:      #D97706;
  --amber-lt:   #FFFBEB;
  --amber-bd:   #FDE68A;
  --t1:         #0B1F3A;
  --t2:         #374151;
  --t3:         #6B7280;
  --t4:         #9CA3AF;
  --r8:  8px;
  --r12: 12px;
  --r16: 16px;
  --r22: 22px;
}

html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--t1) !important;
  background: var(--bg) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem !important; max-width: 1280px !important; }

/* ── Sidebar (collapsed by default, hide trigger) ── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Algorithm radio cards ── */
div[data-testid="stHorizontalBlock"] { gap: 12px !important; }

/* ── All inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r8) !important;
  color: var(--t1) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.92rem !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
  color: var(--t3) !important;
  font-size: 0.75rem !important;
  font-weight: 700 !important;
  letter-spacing: .05em !important;
  text-transform: uppercase !important;
}
div[data-baseweb="select"] > div { border: none !important; background: transparent !important; }

/* ── Select slider ── */
div[data-testid="stSlider"] > div > div > div {
  background: var(--blue) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
  width: 100% !important;
  background: var(--navy) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r12) !important;
  padding: 16px 28px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  letter-spacing: .02em !important;
  box-shadow: 0 4px 16px rgba(11,31,58,.2) !important;
  transition: all .2s ease !important;
}
div[data-testid="stButton"] > button:hover {
  background: var(--navy2) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(11,31,58,.28) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r8) !important;
  font-size: .82rem !important;
  font-weight: 600 !important;
  color: var(--t3) !important;
}
hr { border-color: var(--border) !important; margin: 1.25rem 0 !important; }

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
  from { opacity: 0; transform: scale(.92); }
  to   { opacity: 1; transform: scale(1); }
}
.au  { animation: fadeUp .4s cubic-bezier(.4,0,.2,1) both; }
.au2 { animation: fadeUp .4s .1s cubic-bezier(.4,0,.2,1) both; }
.au3 { animation: fadeUp .4s .18s cubic-bezier(.4,0,.2,1) both; }
.pop { animation: popIn .35s cubic-bezier(.34,1.56,.64,1) both; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    try:
        models = {
            "Random Forest":       joblib.load("models/loan_rf.pkl"),
            "Logistic Regression": joblib.load("models/loan_lr.pkl"),
            "Decision Tree":       joblib.load("models/loan_dt.pkl"),
        }
        encoders        = joblib.load("models/encoders.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        return models, encoders, feature_columns, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

models, encoders, feature_columns, load_error = load_artifacts()

def safe_encode(encoder, col, value):
    if value not in encoder.classes_:
        st.error(f"Unknown value **'{value}'** for **{col}**. Known: {list(encoder.classes_)}")
        st.stop()
    return int(encoder.transform([value])[0])


# ══════════════════════════════════════════════════════
# TOPBAR / HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="au" style="
  background: var(--navy);
  border-radius: var(--r22);
  padding: 28px 36px;
  margin-bottom: 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
">
  <!-- Brand -->
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:46px;height:46px;background:linear-gradient(135deg,#2563EB,#1D4ED8);
         border-radius:13px;display:flex;align-items:center;justify-content:center;
         font-size:1.4rem;flex-shrink:0;">🏦</div>
    <div>
      <div style="font-size:1.3rem;font-weight:800;color:#F1F5F9;line-height:1.1;">LoanIQ</div>
      <div style="font-size:.7rem;color:#64748B;font-weight:600;letter-spacing:.07em;
           text-transform:uppercase;">Credit Risk Predictor</div>
    </div>
  </div>
  <!-- Stats row -->
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
         border-radius:10px;padding:10px 18px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#34D399;">1,000</div>
      <div style="font-size:.65rem;color:#64748B;text-transform:uppercase;
           letter-spacing:.05em;margin-top:1px;">Records</div>
    </div>
    <div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
         border-radius:10px;padding:10px 18px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#60A5FA;">9</div>
      <div style="font-size:.65rem;color:#64748B;text-transform:uppercase;
           letter-spacing:.05em;margin-top:1px;">Features</div>
    </div>
    <div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
         border-radius:10px;padding:10px 18px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#A78BFA;">~76%</div>
      <div style="font-size:.65rem;color:#64748B;text-transform:uppercase;
           letter-spacing:.05em;margin-top:1px;">RF Accuracy</div>
    </div>
    <div style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
         border-radius:10px;padding:10px 18px;text-align:center;">
      <div style="font-size:1.1rem;font-weight:800;color:#FCD34D;">0.78+</div>
      <div style="font-size:.65rem;color:#64748B;text-transform:uppercase;
           letter-spacing:.05em;margin-top:1px;">ROC-AUC</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Error banner ──────────────────────────────────────
if load_error:
    st.markdown(f"""
    <div style="background:#FEF2F2;border:1.5px solid #FECACA;border-radius:var(--r16);
         padding:22px 26px;margin-bottom:24px;">
      <div style="font-weight:800;color:#DC2626;font-size:1.05rem;margin-bottom:8px;">
        ⚠️ Model files not found
      </div>
      <div style="color:#7F1D1D;font-size:.9rem;line-height:1.75;">
        You need to run the notebook first to train and save the models.<br>
        <b>Step 1:</b> Open <code style="background:#FEE2E2;padding:1px 7px;border-radius:4px;">
        loan_prediction.ipynb</code> in Jupyter and click <b>Run All</b>.<br>
        <b>Step 2:</b> Come back and run <code style="background:#FEE2E2;padding:1px 7px;
        border-radius:4px;">python -m streamlit run app.py</code><br><br>
        <span style="opacity:.6;font-size:.82rem;">Error: {load_error}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════
# ALGORITHM SELECTOR (main page, 3 clickable cards)
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="au" style="margin-bottom:8px;">
  <div style="font-size:.72rem;font-weight:700;color:var(--t4);text-transform:uppercase;
       letter-spacing:.08em;margin-bottom:12px;">① Choose Prediction Algorithm</div>
</div>
""", unsafe_allow_html=True)

algo_col1, algo_col2, algo_col3 = st.columns(3, gap="small")

algo_data = {
    "Random Forest": {
        "icon": "🌲",
        "tag": "Best Accuracy",
        "tag_bg": "#ECFDF5", "tag_col": "#059669",
        "desc": "200 trees voting together. Most robust model — handles complex non-linear patterns.",
        "stats": "~76% Acc · 0.79 AUC"
    },
    "Logistic Regression": {
        "icon": "📈",
        "tag": "Most Interpretable",
        "tag_bg": "#EEF3FF", "tag_col": "#2563EB",
        "desc": "Linear model with StandardScaler pipeline. Fast, transparent, and reliable baseline.",
        "stats": "~73% Acc · 0.78 AUC"
    },
    "Decision Tree": {
        "icon": "🌳",
        "tag": "Easiest to Explain",
        "tag_bg": "#FFFBEB", "tag_col": "#D97706",
        "desc": "Single tree, max depth 6. Clear decision rules — easy to trace each prediction.",
        "stats": "~70% Acc · 0.64 AUC"
    },
}

# Use session state to track selected algorithm
if "selected_algo" not in st.session_state:
    st.session_state.selected_algo = "Random Forest"

for col, (name, info) in zip([algo_col1, algo_col2, algo_col3], algo_data.items()):
    with col:
        is_selected = st.session_state.selected_algo == name
        border_col  = "#2563EB" if is_selected else "var(--border)"
        bg_col      = "#F0F5FF" if is_selected else "var(--surface)"
        shadow      = "0 4px 18px rgba(37,99,235,.15)" if is_selected else "0 1px 4px rgba(0,0,0,.04)"
        check       = f"""<div style="position:absolute;top:12px;right:12px;width:20px;height:20px;
                          background:#2563EB;border-radius:50%;display:flex;align-items:center;
                          justify-content:center;font-size:.7rem;color:white;">✓</div>""" if is_selected else ""

        st.markdown(f"""
        <div style="background:{bg_col};border:2px solid {border_col};border-radius:var(--r16);
             padding:20px 18px 16px;position:relative;box-shadow:{shadow};
             min-height:148px;transition:all .2s ease;">
          {check}
          <div style="font-size:1.6rem;margin-bottom:8px;">{info['icon']}</div>
          <div style="font-size:.92rem;font-weight:700;color:var(--t1);margin-bottom:6px;">
            {name}
          </div>
          <div style="display:inline-block;background:{info['tag_bg']};color:{info['tag_col']};
               border-radius:20px;padding:2px 10px;font-size:.65rem;font-weight:700;
               text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">
            {info['tag']}
          </div>
          <div style="font-size:.78rem;color:var(--t3);line-height:1.6;margin-bottom:8px;">
            {info['desc']}
          </div>
          <div style="font-size:.72rem;color:var(--t4);font-weight:600;">{info['stats']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Select", key=f"btn_{name}", use_container_width=True):
            st.session_state.selected_algo = name
            st.rerun()

selected_model_name = st.session_state.selected_algo
model = models[selected_model_name]

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# INPUT + RESULTS (two columns)
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="au2" style="font-size:.72rem;font-weight:700;color:var(--t4);
     text-transform:uppercase;letter-spacing:.08em;margin-bottom:20px;">
  ② Enter Customer Details &amp; Predict
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

purpose_map = {
    "car":                "🚗 Car",
    "furniture/equipment":"🛋️ Furniture / Equipment",
    "radio/TV":           "📺 Radio / TV",
    "domestic appliances":"🏠 Domestic Appliances",
    "repairs":            "🔧 Repairs",
    "education":          "🎓 Education",
    "business":           "💼 Business",
    "vacation/others":    "✈️ Vacation / Others",
}

# ── LEFT: Form ────────────────────────────────────────
with left:

    # Personal
    st.markdown("""
    <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px;">
      <div style="width:30px;height:30px;background:var(--blue-lt);border-radius:8px;
           display:flex;align-items:center;justify-content:center;font-size:.95rem;">👤</div>
      <div>
        <div style="font-weight:700;font-size:.93rem;color:var(--t1);">Personal Details</div>
        <div style="font-size:.72rem;color:var(--t4);">Demographic information</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
    with p2:
        sex = st.selectbox("Sex", ["male","female"],
                           format_func=lambda x: "♂ Male" if x=="male" else "♀ Female")
    with p3:
        housing = st.selectbox("Housing", ["own","free","rent"],
                               format_func=lambda x: {"own":"🏠 Own","free":"🏡 Free","rent":"🏢 Rent"}[x])

    job = st.select_slider(
        "Employment Level",
        options=[0,1,2,3], value=2,
        format_func=lambda x: {
            0:"0 · Unskilled (non-resident)",
            1:"1 · Unskilled (resident)",
            2:"2 · Skilled",
            3:"3 · Highly Skilled"
        }[x]
    )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # Financial
    st.markdown("""
    <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px;">
      <div style="width:30px;height:30px;background:var(--green-lt);border-radius:8px;
           display:flex;align-items:center;justify-content:center;font-size:.95rem;">💳</div>
      <div>
        <div style="font-weight:700;font-size:.93rem;color:var(--t1);">Financial Profile</div>
        <div style="font-size:.72rem;color:var(--t4);">Accounts and credit terms</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        saving_acc = st.selectbox("Saving Accounts",
            ["none","little","moderate","quite rich","rich"],
            format_func=lambda x:{
                "none":"⬜ None","little":"🟡 Little",
                "moderate":"🟠 Moderate","quite rich":"🟢 Quite Rich","rich":"💚 Rich"
            }[x])
    with f2:
        checking_acc = st.selectbox("Checking Account",
            ["none","little","moderate","rich"],
            format_func=lambda x:{
                "none":"⬜ None","little":"🟡 Little",
                "moderate":"🟠 Moderate","rich":"💚 Rich"
            }[x])

    g1, g2 = st.columns(2)
    with g1:
        credit_amount = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000,
                                        value=3000, step=250)
    with g2:
        duration = st.number_input("Duration (months)", min_value=4, max_value=72,
                                   value=18, step=1)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    # Purpose
    st.markdown("""
    <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px;">
      <div style="width:30px;height:30px;background:var(--amber-lt);border-radius:8px;
           display:flex;align-items:center;justify-content:center;font-size:.95rem;">🎯</div>
      <div>
        <div style="font-weight:700;font-size:.93rem;color:var(--t1);">Loan Purpose</div>
        <div style="font-size:.72rem;color:var(--t4);">What will this credit be used for?</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    purpose = st.selectbox("Purpose", list(purpose_map.keys()),
                           format_func=lambda x: purpose_map[x],
                           label_visibility="collapsed")

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Analyse Credit Risk", use_container_width=True)


# ── RIGHT: Results ────────────────────────────────────
with right:

    if not predict_clicked:
        st.markdown(f"""
        <div style="background:var(--surface);border:1.5px dashed var(--border);
             border-radius:var(--r22);padding:44px 28px;text-align:center;margin-bottom:18px;">
          <div style="font-size:2.6rem;margin-bottom:12px;">🏦</div>
          <div style="font-size:1.05rem;font-weight:700;color:var(--t1);margin-bottom:8px;">
            Awaiting Assessment
          </div>
          <div style="font-size:.875rem;color:var(--t4);line-height:1.75;
               max-width:260px;margin:0 auto;">
            Selected model: <b style="color:var(--navy);">{selected_model_name}</b><br>
            Fill in the form and click <b style="color:var(--navy);">Analyse Credit Risk</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        for bg, bd, col, icon, title, desc in [
            ("var(--green-lt)","var(--green-bd)","#065F46","✅",
             "Low Risk Profile",
             "Rich savings · Checking balance · Short loan duration · Skilled employment"),
            ("var(--red-lt)","var(--red-bd)","#7F1D1D","⚠️",
             "High Risk Profile",
             "No savings · No checking · High amount · Long duration · Unskilled job"),
        ]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bd};border-radius:var(--r12);
                 padding:15px 18px;margin-bottom:10px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <span>{icon}</span>
                <span style="font-weight:700;color:{col};font-size:.87rem;">{title}</span>
              </div>
              <div style="font-size:.78rem;color:var(--t3);line-height:1.6;padding-left:22px;">
                {desc}
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Encode
        sex_enc      = safe_encode(encoders["Sex"],              "Sex",              sex)
        housing_enc  = safe_encode(encoders["Housing"],          "Housing",          housing)
        saving_enc   = safe_encode(encoders["Saving accounts"],  "Saving accounts",  saving_acc)
        checking_enc = safe_encode(encoders["Checking account"], "Checking account", checking_acc)
        purpose_enc  = safe_encode(encoders["Purpose"],          "Purpose",          purpose)

        input_dict = {
            "Age": age, "Sex": sex_enc, "Job": job,
            "Housing": housing_enc, "Saving accounts": saving_enc,
            "Checking account": checking_enc, "Credit amount": credit_amount,
            "Duration": duration, "Purpose": purpose_enc,
        }
        input_df = pd.DataFrame([input_dict])[feature_columns]

        prediction  = model.predict(input_df)[0]
        proba       = model.predict_proba(input_df)[0]
        app_pct     = round(float(proba[0]) * 100, 1)
        rej_pct     = round(float(proba[1]) * 100, 1)
        confidence  = max(app_pct, rej_pct)
        approved    = prediction == 0

        if approved:
            r_bg="#ECFDF5"; r_bd="#6EE7B7"; r_col="#065F46"
            icon="✅"; label="Loan Approved"; sub="Good Credit Risk"; c_col="#059669"
        else:
            r_bg="#FEF2F2"; r_bd="#FCA5A5"; r_col="#7F1D1D"
            icon="❌"; label="Loan Rejected";  sub="Bad Credit Risk";  c_col="#DC2626"

        # Result card
        st.markdown(f"""
        <div class="pop" style="background:{r_bg};border:2px solid {r_bd};
             border-radius:var(--r22);padding:28px 24px;text-align:center;
             margin-bottom:16px;box-shadow:0 6px 24px rgba(0,0,0,.07);">
          <div style="font-size:2.6rem;margin-bottom:8px;">{icon}</div>
          <div style="font-size:1.65rem;font-weight:800;color:{r_col};
               line-height:1.15;margin-bottom:4px;">{label}</div>
          <div style="font-size:.86rem;color:{r_col};opacity:.72;font-weight:500;
               margin-bottom:16px;">{sub} · {selected_model_name}</div>
          <div style="display:inline-flex;align-items:center;gap:8px;
               background:rgba(255,255,255,.75);border-radius:10px;padding:8px 18px;">
            <span style="font-size:.7rem;color:var(--t3);font-weight:700;
                  text-transform:uppercase;letter-spacing:.06em;">Confidence</span>
            <span style="font-size:1.05rem;font-weight:800;color:{c_col};">{confidence}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability tiles
        t1c, t2c = st.columns(2)
        for tc, pct, lbl, color, bg in [
            (t1c, app_pct, "Approval",  "#059669","#ECFDF5"),
            (t2c, rej_pct, "Rejection", "#DC2626","#FEF2F2"),
        ]:
            with tc:
                st.markdown(f"""
                <div style="background:{bg};border-radius:var(--r12);
                     padding:16px 18px;text-align:center;margin-bottom:4px;">
                  <div style="font-size:.68rem;color:var(--t3);font-weight:700;
                       text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;">
                    {lbl}
                  </div>
                  <div style="font-size:2rem;font-weight:800;color:{color};line-height:1;">
                    {pct}%
                  </div>
                  <div style="background:#E5E7EB;border-radius:4px;height:5px;
                       margin-top:10px;overflow:hidden;">
                    <div style="background:{color};height:100%;width:{pct}%;
                         border-radius:4px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Plotly chart
        fig = go.Figure(go.Bar(
            x=["Approval", "Rejection"],
            y=[app_pct, rej_pct],
            marker=dict(color=["#059669","#DC2626"], line=dict(width=0)),
            text=[f"<b>{app_pct}%</b>", f"<b>{rej_pct}%</b>"],
            textposition="outside",
            textfont=dict(family="Plus Jakarta Sans", size=13,
                          color=["#059669","#DC2626"]),
            hoverinfo="skip", width=0.4,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(244,246,251,1)",
            height=175,
            margin=dict(l=4, r=4, t=14, b=4),
            yaxis=dict(range=[0,118], showgrid=False,
                       showticklabels=False, zeroline=False),
            xaxis=dict(showgrid=False,
                       tickfont=dict(family="Plus Jakarta Sans",
                                     size=11, color="#6B7280")),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Summary strip
        st.markdown(f"""
        <div style="background:var(--surface2);border:1px solid var(--border);
             border-radius:var(--r12);padding:12px 18px;display:flex;flex-wrap:wrap;
             gap:14px;align-items:center;margin-bottom:12px;">
          <span style="font-size:.8rem;color:var(--t3);">
            <b style="color:var(--t1);">Model:</b> {selected_model_name}
          </span>
          <span style="font-size:.8rem;color:var(--t3);">
            <b style="color:var(--t1);">Amount:</b> DM {credit_amount:,}
          </span>
          <span style="font-size:.8rem;color:var(--t3);">
            <b style="color:var(--t1);">Duration:</b> {duration} months
          </span>
          <span style="font-size:.8rem;color:var(--t3);">
            <b style="color:var(--t1);">Purpose:</b> {purpose_map[purpose]}
          </span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔎  View encoded feature vector sent to model"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════
st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid var(--border);padding-top:18px;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
  <div style="font-size:.75rem;color:var(--t4);">
    🏦 <b style="color:var(--t3);">LoanIQ</b> ·
    German Credit Dataset · Logistic Regression · Decision Tree · Random Forest
  </div>
  <div style="font-size:.73rem;color:var(--t4);">
    Built by <b style="color:var(--t3);">Adil Khan</b> ·
    scikit-learn · Streamlit · Plotly
  </div>
</div>
""", unsafe_allow_html=True)
