import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv("D:\\archive\\creditcard.csv")

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# Create Streamlit app
st.title("💳 Credit Card Fraud Detection Model")
st.write(f"**Model Accuracy:**")
st.write(f"✅ **Training Accuracy:** {train_acc:.2f}")
st.write(f"✅ **Testing Accuracy:** {test_acc:.2f}")
st.write("### Enter the following features to check if the transaction is legitimate or fraudulent:")

# Input field for user to enter feature values
input_df = st.text_input('Enter feature values (comma-separated):')

# Button to submit input and get prediction
submit = st.button("🔍 Predict")

if submit:
    try:
        # Process input: Remove whitespace, replace tabs, and convert to float
        input_df_lst = [x.strip().replace('\t', '') for x in input_df.split(',')]
        features = np.array(input_df_lst, dtype=np.float64)

        # Make prediction
        prediction = model.predict(features.reshape(1, -1))

        # Display result
        if prediction[0] == 0:
            st.success("✅ Legitimate transaction")
        else:
            st.error("🚨 Fraudulent transaction detected!")

    except ValueError as e:
        st.error(f"Invalid input format: {e}")
        st.info("Please enter **comma-separated numeric values**.")
