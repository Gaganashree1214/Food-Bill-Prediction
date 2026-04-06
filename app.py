import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Bill Predictor",
    page_icon="🍔",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #0f0e11;
    color: #f0ede8;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a1820 0%, #201c2a 100%);
    border: 1px solid #2e2a3a;
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #ff6b3520 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, #a855f720 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: #ff6b3518;
    border: 1px solid #ff6b3540;
    color: #ff6b35;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #f0ede8;
    margin: 0 0 0.6rem 0;
    line-height: 1.1;
}
.hero h1 span { color: #ff6b35; }
.hero p {
    color: #9b96a8;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
    max-width: 480px;
    line-height: 1.6;
}

/* ── Section heading ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #ff6b35;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #2e2a3a;
}

/* ── Form card ── */
.form-card {
    background: #1a1820;
    border: 1px solid #2e2a3a;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #9b96a8 !important;
    margin-bottom: 4px !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #0f0e11 !important;
    border: 1px solid #2e2a3a !important;
    border-radius: 10px !important;
    color: #f0ede8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="stNumberInput"] input:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 2px #ff6b3518 !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #ff6b35, #e8421a) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.85rem 2rem !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    width: 100% !important;
    margin-top: 1rem !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px #ff6b3540 !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #1e1028, #1a1820);
    border: 1px solid #a855f740;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    text-align: center;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #a855f7, #ff6b35, transparent);
}
.result-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9b96a8;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.result-sub {
    color: #9b96a8;
    font-size: 13px;
    margin-top: 0.5rem;
}

/* ── Summary table ── */
.summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.summary-table tr { border-bottom: 1px solid #2e2a3a; }
.summary-table tr:last-child { border-bottom: none; }
.summary-table td { padding: 10px 6px; }
.summary-table td:first-child { color: #9b96a8; font-size: 12px; letter-spacing: 0.5px; }
.summary-table td:last-child { color: #f0ede8; font-weight: 500; text-align: right; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a1820 !important;
    border-right: 1px solid #2e2a3a !important;
}
section[data-testid="stSidebar"] * { color: #f0ede8 !important; }
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #f0ede8;
    margin-bottom: 0.3rem;
}
.sidebar-logo span { color: #ff6b35; }
.stat-pill {
    background: #0f0e11;
    border: 1px solid #2e2a3a;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-pill-label { font-size: 11px; color: #9b96a8; letter-spacing: 1px; text-transform: uppercase; }
.stat-pill-value { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 15px; color: #f0ede8; }

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: #1a1820 !important;
    border: 1px solid #2e2a3a !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: #9b96a8 !important;
    font-size: 13px !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    background: #1a1820 !important;
    border-radius: 12px !important;
    border: 1px solid #2e2a3a !important;
}

/* ── Divider ── */
hr { border-color: #2e2a3a !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
RESTAURANT_NAMES = ['Spice Garden', 'Pizza Palace', 'Burger Barn', 'Sushi Stop', 'Green Bowl', 'Taco Town', 'Noodle Nest']
CUISINE_TYPES    = ['indian', 'italian', 'american', 'japanese', 'healthy', 'mexican', 'chinese']
MEAL_TIMES       = ['lunch', 'dinner', 'breakfast']

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        return None, None
    model = joblib.load("model.pkl")
    model_cols = list(model.feature_names_in_)
    return model, model_cols

model, model_cols = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Bill<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9b96a8;font-size:12px;margin-bottom:1.5rem;">Food Delivery Predictor</p>', unsafe_allow_html=True)
    st.markdown("---")

    if model is not None:
        st.markdown('<p style="color:#4ade80;font-size:12px;font-weight:500;">● Model loaded</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-pill"><span class="stat-pill-label">Algorithm</span><span class="stat-pill-value">Linear Reg.</span></div>
        <div class="stat-pill"><span class="stat-pill-label">Features</span><span class="stat-pill-value">{len(model_cols)}</span></div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p style="color:#9b96a8;font-size:12px;line-height:1.6;">Model trained in Jupyter notebook. To retrain, run your <code>.ipynb</code> and call <code>joblib.dump(model, "model.pkl")</code></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#f87171;font-size:12px;font-weight:500;">● model.pkl not found</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0f0e11;border:1px solid #2e2a3a;border-radius:10px;padding:14px;font-size:12px;color:#9b96a8;line-height:1.8;">
        Run your notebook and save:<br>
        <code style="color:#ff6b35;">joblib.dump(model, 'model.pkl')</code><br><br>
        Place <code>model.pkl</code> in the same folder as <code>app.py</code>.
        </div>
        """, unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">ML-Powered Prediction</div>
    <h1>Food Delivery<br><span>Bill Estimator</span></h1>
    <p>Enter your order details below and get an instant price prediction powered by a trained Linear Regression model.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ No trained model found. Please run your notebook to generate `model.pkl`, then restart the app.")
    st.stop()

# ─── Form ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Order Details</div>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        restaurant_name = st.selectbox("Restaurant Name",  RESTAURANT_NAMES)
        cuisine_type    = st.selectbox("Cuisine Type",     CUISINE_TYPES)
        meal_time       = st.selectbox("Meal Time",        MEAL_TIMES)
        customer_gender = st.selectbox("Customer Gender",  ["male", "female"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        num_items         = st.number_input("Number of Items",        min_value=1,   max_value=20,   value=3,   step=1)
        avg_item_price    = st.number_input("Avg Item Price (₹)",     min_value=10,  max_value=999,  value=200, step=10)
        discount_percent  = st.number_input("Discount (%)",           min_value=0,   max_value=100,  value=10,  step=5)
        delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        delivery_rating = st.number_input("Delivery Rating (1–5)", min_value=1.0, max_value=5.0,  value=4.5, step=0.1)
        customer_age    = st.number_input("Customer Age",           min_value=18,  max_value=100,  value=28,  step=1)
        weekend         = st.selectbox("Is Weekend?",               ["Yes", "No"])
        num_prev_orders = st.number_input("Previous Orders",        min_value=0,   max_value=100,  value=5,   step=1)
        st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("Predict Total Bill →", use_container_width=True)

# ─── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    gender_enc  = 1 if customer_gender == "male" else 0
    weekend_enc = 1 if weekend == "Yes" else 0

    total_items_price = num_items * avg_item_price
    discount_amount   = total_items_price * (discount_percent / 100)

    input_row = {col: 0 for col in model_cols}
    input_row['num_items']            = num_items
    input_row['avg_item_price']       = avg_item_price
    input_row['discount_percent']     = discount_percent
    input_row['delivery_distance_km'] = delivery_distance
    input_row['delivery_rating']      = delivery_rating
    input_row['customer_age']         = customer_age
    input_row['customer_gender']      = gender_enc
    input_row['weekend']              = weekend_enc
    input_row['num_previous_orders']  = num_prev_orders
    input_row['total_items_price']    = total_items_price
    input_row['discount_amount']      = discount_amount

    rest_col = f"restaurant_name_{restaurant_name}"
    if rest_col in input_row:
        input_row[rest_col] = 1

    cuis_col = f"cuisine_type_{cuisine_type}"
    if cuis_col in input_row:
        input_row[cuis_col] = 1

    meal_col = f"meal_time_{meal_time}"
    if meal_col in input_row:
        input_row[meal_col] = 1

    input_df   = pd.DataFrame([input_row])
    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Total Bill</div>
        <div class="result-amount">₹{prediction:,.2f}</div>
        <div class="result-sub">{restaurant_name} · {cuisine_type.title()} · {meal_time.title()}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View full input breakdown"):
        rows = [
            ("Restaurant",       restaurant_name),
            ("Cuisine",          cuisine_type.title()),
            ("Meal Time",        meal_time.title()),
            ("Items",            str(num_items)),
            ("Avg Item Price",   f"₹{avg_item_price:,}"),
            ("Discount",         f"{discount_percent}%"),
            ("Distance",         f"{delivery_distance} km"),
            ("Delivery Rating",  str(delivery_rating)),
            ("Customer Age",     str(customer_age)),
            ("Gender",           customer_gender.title()),
            ("Weekend",          weekend),
            ("Previous Orders",  str(num_prev_orders)),
            ("Total Items Price",f"₹{total_items_price:,.2f}"),
            ("Discount Amount",  f"₹{discount_amount:,.2f}"),
            ("Predicted Bill",   f"₹{prediction:,.2f}"),
        ]
        rows_html = "".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>"
            for label, value in rows
        )
        st.markdown(f'<table class="summary-table"><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)