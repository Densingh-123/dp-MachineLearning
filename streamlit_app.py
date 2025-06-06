import streamlit as st
import pandas as pd  # <-- Fix here

st.title('🤖 Machine Learning App')

st.info("This is an app to build a Machine Learning Model.")

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

# Show the dataset
st.write(df)
