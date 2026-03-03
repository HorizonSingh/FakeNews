import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a news article below and check if it's Real or Fake.")

# Text input box
user_input = st.text_area("Enter News Article Here:", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        
        # Convert input text to numerical form
        transformed_input = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(transformed_input)
        
        # Get confidence
        probability = model.predict_proba(transformed_input)

        if prediction[0] == "FAKE":
            st.error("🚨 This looks like FAKE news!")
        else:
            st.success("✅ This looks like REAL news!")

        st.write(f"Confidence Score: {round(max(probability[0]) * 100, 2)}%")

    else:
        st.warning("Please enter some text before predicting.")
