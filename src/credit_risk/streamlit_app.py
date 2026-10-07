"""Small Streamlit front end for the trained model."""
import streamlit as st
from credit_risk.pipeline import score_application
st.set_page_config(page_title="Credit Risk", page_icon="??")
st.title("Credit-risk decision demo")
st.caption("Synthetic fallback data; not for production lending decisions.")
with st.form("application"):
    income = st.number_input("Annual income", 18000.0, 250000.0, 60000.0)
    debt_ratio = st.slider("Debt ratio", 0.01, 0.98, 0.30)
    history = st.number_input("Credit history (years)", 0.0, 45.0, 8.0)
    loan = st.number_input("Loan amount", 1000.0, 100000.0, 15000.0)
    late = st.number_input("Late payments", 0, 20, 1)
    employment = st.number_input("Employment years", 0.0, 40.0, 5.0)
    age = st.number_input("Age", 21, 75, 35)
    age_group = st.selectbox("Age group", ["young", "middle", "older"])
    home = st.selectbox("Home ownership", ["rent", "own", "mortgage"])
    submitted = st.form_submit_button("Score")
if submitted:
    try:
        result = score_application(locals())
        st.metric("Default probability", f"{result['default_probability']:.1%}")
        st.warning(result["decision"])
    except FileNotFoundError: st.error("Train the model first: risk-train")
