import streamlit as st
import pandas as pd

df = pd.read_csv("outputs/results.csv")

st.title("SLA Dashboard")

safe = (df['predicted_breach']==0).sum()
unsafe = (df['predicted_breach']==1).sum()

st.metric("SAFE", safe)
st.metric("UNSAFE", unsafe)

st.dataframe(df[df['predicted_breach']==1])