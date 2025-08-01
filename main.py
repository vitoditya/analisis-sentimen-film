# streamlit_app.py

import streamlit as st
import re
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model

from sklearn.feature_extraction.text import TfidfVectorizer

# Download resource NLTK jika belum ada
nltk.download('punkt')
nltk.download('stopwords')

# --- Fungsi Preprocessing ---
def preprocess(text):
    # Cleaning dan Case Folding
    text = re.sub(r'<.*?>', '', text)  # Hapus tag HTML
    text = re.sub(r"[^a-zA-Z']", ' ', text).lower()
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Stopword Removal dan Stemming
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# --- Load Model dan Vectorizer ---
model_nb = joblib.load("naive_bayes_model.pkl")
model_svm = joblib.load("svm_model.pkl")
model_cnn = load_model("cnn_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- UI Streamlit ---
st.set_page_config(page_title="Analisis Sentimen Film", layout="centered")
st.title("🎬 Analisis Sentimen Ulasan Film")
st.markdown("Masukkan ulasan film, pilih model yang ingin kamu gunakan, dan lihat hasil prediksinya.")

text_input = st.text_area("Masukkan ulasan film:")
model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "SVM", "CNN"])

if st.button("Analisis"):
    if not text_input.strip():
        st.warning("Teks ulasan tidak boleh kosong.")
    else:
        # Preprocessing
        cleaned = preprocess(text_input)
        vectorized = vectorizer.transform([cleaned])

        # Prediksi
        if model_choice == "Naive Bayes":
            result = model_nb.predict(vectorized)[0]
        elif model_choice == "SVM":
            result = model_svm.predict(vectorized)[0]
        else:  # CNN
            result = model_cnn.predict(vectorized.toarray())
            result = np.argmax(result, axis=1)[0]  # Ambil kelas dengan skor tertinggi

        label = "Positif" if result == 1 else "Negatif"
        st.success(f"Hasil Sentimen: **{label}**")
