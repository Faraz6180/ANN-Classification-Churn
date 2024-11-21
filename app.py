import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set custom styles for Streamlit
st.markdown(
    """
    <style>
    .main {
        background-color: #1a1a2e;
        color: #eaeaea;
    }
    .sidebar .sidebar-content {
        background-color: #0f3460;
        color: #eaeaea;
    }
    .stButton > button {
        background-color: #e94560;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background-color: #16213e;
        color: white;
    }
    .stSelectbox > div > div > div > button {
        background-color: #16213e;
        color: white;
    }
    .report-container {
        background-color: #16213e;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 10px #0f3460;
        margin-bottom: 25px;
    }
    .report-title {
        font-size: 24px;
        font-weight: bold;
        color: #e94560;
        margin-bottom: 20px;
    }
    .chart-section {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title
st.title('📊 Customer Churn Prediction')
st.write("Welcome to the enhanced **Customer Churn Prediction** tool. Use this application to predict customer churn and explore factors that might influence customer retention!")

# User input section
st.sidebar.header('Customer Details')
st.sidebar.subheader("Fill in the customer's information below:")

# User Inputs
geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('🧍‍ Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🔢 Age', 18, 92)
balance = st.sidebar.number_input('💰 Balance', min_value=0.0, step=0.01)
credit_score = st.sidebar.number_input('📊 Credit Score', min_value=300, max_value=850)
estimated_salary = st.sidebar.number_input('💵 Estimated Salary', min_value=0.0, step=0.01)
tenure = st.sidebar.slider('📅 Tenure (years)', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('⚡ Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])  # Convert to a standard Python float

# Function to plot a pie chart for churn probability
def plot_churn_pie(probability):
    labels = ['Will Not Churn', 'Will Churn']
    sizes = [1 - probability, probability]
    colors = ['#28a745', '#e94560']
    explode = (0, 0.1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')
    return fig

# Function to visualize reasons for churn
def plot_reasons(reasons):
    labels = list(reasons.keys())
    values = list(reasons.values())
    colors = ['#ff6f61' if v == 1 else '#6abf69' for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel('Risk Indicators')
    ax.set_title('Factors Contributing to Churn')
    plt.tight_layout()
    return fig

# Display prediction results with a pie chart
st.markdown("<div class='report-container'><div class='report-title'>Churn Probability</div></div>", unsafe_allow_html=True)
st.write(f'### Churn Probability: **{prediction_proba * 100:.2f}%**')

# Visualize churn probability
fig_pie = plot_churn_pie(prediction_proba)
st.pyplot(fig_pie)

# Display reasons behind churn and suggestions
st.markdown("<div class='report-container'><div class='report-title'>🔍 Reasons Behind Prediction & Retention Suggestions</div></div>", unsafe_allow_html=True)

# Analyze reasons for churn
reasons = {
    "Low Credit Score": int(credit_score < 500),
    "Low Balance": int(balance < 1000),
    "Inactive Member": int(not is_active_member),
    "Multiple Products": int(num_of_products > 2)
}

# Display visual reasons for churn
fig_reasons = plot_reasons(reasons)
st.pyplot(fig_reasons)

# Display individual reasons and suggestions
for reason, triggered in reasons.items():
    if triggered:
        st.error(f"⚠️ {reason} is a risk factor.")
    else:
        st.success(f"✅ {reason} is not a concern.")

# Recommendations for user action
st.markdown("<div class='report-container'><div class='report-title'>✅ Suggestions to Retain the Customer:</div></div>", unsafe_allow_html=True)
if credit_score < 500:
    st.info("🔹 **Improve Credit Score**: Encourage timely payments and proper credit usage.")
if balance < 1000:
    st.info("🔹 **Increase Account Balance**: Educate on benefits of higher savings.")
if not is_active_member:
    st.info("🔹 **Engage Inactive Members**: Introduce loyalty programs and rewards.")
if num_of_products > 2:
    st.info("🔹 **Simplify Product Portfolio**: Offer consolidated packages.")
