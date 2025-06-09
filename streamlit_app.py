import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')
st.info('This app builds a machine learning model to predict penguin species!')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

df = load_data()

# Display data
with st.expander('📊 Data'):
    st.write('**Raw data**')
    st.dataframe(df)

    st.write('**X (features)**')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**y (target)**')
    y_raw = df.species
    st.write(y_raw)

# Data visualization
with st.expander('📈 Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar input
with st.sidebar:
    st.header('🧮 Input features')
    island = st.selectbox('Island', df['island'].unique())
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ['male', 'female'])

    input_df = pd.DataFrame({
        'island': [island],
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [gender]
    })

# Combine user input with dataset
input_penguins = pd.concat([input_df, X_raw], axis=0).reset_index(drop=True)

with st.expander('🧾 Input Data'):
    st.write('**User input**')
    st.dataframe(input_df)

    st.write('**Combined input with full dataset**')
    st.dataframe(input_penguins)

# Data preprocessing
df_encoded = pd.get_dummies(input_penguins, columns=['island', 'sex'])
input_row = df_encoded.iloc[[0]]
X = df_encoded.iloc[1:]

# Encode y labels
y = y_raw.map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

with st.expander('⚙️ Encoded Data'):
    st.write('**Input features after encoding**')
    st.dataframe(input_row)

    st.write('**Target variable (y)**')
    st.write(y)

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Predict
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Prepare prediction DataFrame
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Output
st.subheader('🔍 Prediction Probability')
st.dataframe(df_prediction_proba.style.format("{:.2%}"), hide_index=True)

# Final prediction
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"🧬 Predicted Species: **{penguins_species[prediction][0]}**")
