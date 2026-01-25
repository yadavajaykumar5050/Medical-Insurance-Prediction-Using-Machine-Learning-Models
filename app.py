import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------- BLACK BACKGROUND CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }

    label, p, span, div {
        color: white !important;
    }

    input {
        background-color: #1e1e1e !important;
        color: white !important;
    }

    button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ----------------------------------------

# Load dataset
medical_df = pd.read_csv('insurance.csv')

# Encoding
medical_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_df.replace({'region': {
    'southeast': 0,
    'southwest': 1,
    'northwest': 2,
    'northeast': 3
}}, inplace=True)

# Split data
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=2
)

# Train model
lg = LinearRegression()
lg.fit(X_train, y_train)

# Streamlit UI
st.title("💊 Medical Insurance Cost Prediction By Ajay💊")

st.markdown("""
*Enter values in this order (comma separated):*

age, sex, bmi, children, smoker, region

*Encoding Info:*
- sex: male = 0, female = 1  
- smoker: yes = 0, no = 1  
- region: southeast=0, southwest=1, northwest=2, northeast=3  
""")

input_text = st.text_input("Example: 35,0,26.5,2,1,3")

if st.button("Predict Insurance Cost"):
    try:
        input_list = input_text.split(",")

        if len(input_list) != 6:
            st.error("❌ Please enter exactly 6 values.")
        else:
            input_array = np.asarray(input_list, dtype=float)
            prediction = lg.predict(input_array.reshape(1, -1))
            st.success(f"💰 Predicted Insurance Cost: ₹ {prediction[0]:,.2f}")

    except ValueError:
        st.error("❌ Please enter only numeric values.")