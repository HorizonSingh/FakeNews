import streamlit as st
import pickle

# -----------------------------
# Load trained model & vectorizer
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a news article below and check whether it is REAL or FAKE.")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter News Article Here:", height=200)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Convert text to numerical form
        transformed_input = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(transformed_input)
        probabilities = model.predict_proba(transformed_input)

        # Get class labels
        classes = model.classes_

        # Find confidence
        confidence = round(max(probabilities[0]) * 100, 2)

        # Display Result
        if prediction[0] == "0":
            st.error(f"🚨✅ This looks like REAL news!\n\nConfidence: {confidence}%")
        else:
            st.success(f"🚨 This looks like FAKE news!\n\nConfidence: {confidence}%")
