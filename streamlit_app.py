import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info("This is an app to build a Machine Learning Model.")

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

# Data display section
with st.expander('📊 View Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

    st.write('**X (Features)**')
    x = df.drop('species', axis=1)
    st.dataframe(x)

    st.write('**Y (Target / Labels)**')
    y = df['species']
    st.dataframe(y)
with st.expander('Data Visulaisation'):
    st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')
