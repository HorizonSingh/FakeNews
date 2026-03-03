import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter News Article Here:")

if st.button("Predict"):
    if user_input.strip() != "":
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        if prediction[0] == 0:
            st.error("🚨 This looks like FAKE news!")
        else:
            st.success("✅ This looks like REAL news!")
    else:
        st.warning("Please enter some text.")
