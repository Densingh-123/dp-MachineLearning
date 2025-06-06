import streamlit as st
import pandas as np

st.title('🤖 Machine Learning App')

st.info("This is a app build Machine Learning Model..")
df = pd.read_cdv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
df
