# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:00:32 2025

@author: dsand
"""

import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load trained model and vectorizer
model = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Streamlit App
st.set_page_config(page_title="Climate Text Classifier", layout="centered")
st.title("🌍 Climate Text Classifier")
st.write("Enter climate-related text and get predictions using a trained Random Forest model.")

# User Input
user_input = st.text_area("✏️ Enter Text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]
        st.success(f"✅ Predicted Class: **{prediction}**")
