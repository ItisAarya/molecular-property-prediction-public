import streamlit as st, requests
st.title("Molecular Property Predictor (Hybrid)")
smi = st.text_input("SMILES", "CCO")
ds = st.selectbox("Dataset", ["tox21","bbbp","clintox","esol","lipophilicity"])
if st.button("Predict"):
    r = requests.post("http://127.0.0.1:8000/predict", json={"dataset": ds, "smiles": smi})
    st.json(r.json())
