import streamlit as st
import numpy as np
import re
import pickle
import joblib
import nltk
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --- Setup ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tokenizer_treebank = TreebankWordTokenizer()

MAXLEN = 250  # Sesuai training CNN

# --- Preprocessing TF-IDF ---
def preprocess_tfidf(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    text = text.lower()
    tokens = tokenizer_treebank.tokenize(text)
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# --- Preprocessing CNN ---
def preprocess_cnn(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# --- Load Models ---
try:
    model_nb = joblib.load("nb_pipeline.pkl")
    model_svm = joblib.load("svm_pipeline.pkl")
except Exception as e:
    st.error(f"Gagal memuat model Naive Bayes / SVM: {e}")

try:
    model_cnn = load_model("cnn_model.h5")
    with open("cnn_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Gagal memuat model CNN: {e}")

# --- Streamlit UI ---
st.set_page_config(page_title="Analisis Sentimen Film", layout="centered")
st.title("🎬 Analisis Sentimen Ulasan Film")
st.markdown("Masukkan ulasan film, pilih model, dan lihat hasil prediksi sentimennya.")

text_input = st.text_area("Masukkan ulasan film:")
model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "SVM", "CNN"])

col1, col2 = st.columns(2)

with col1:
    if st.button("Analisis"):
        if not text_input.strip():
            st.warning("Teks ulasan tidak boleh kosong.")
        else:
            try:
                if model_choice == "Naive Bayes":
                    cleaned = preprocess_tfidf(text_input)
                    result = model_nb.predict([cleaned])[0]

                elif model_choice == "SVM":
                    cleaned = preprocess_tfidf(text_input)
                    result = model_svm.predict([cleaned])[0]

                else:  # CNN
                    cleaned = preprocess_cnn(text_input)
                    sequence = tokenizer.texts_to_sequences([cleaned])
                    padded = pad_sequences(sequence, maxlen=MAXLEN)
                    pred_prob = model_cnn.predict(padded)[0][0]
                    result = 1 if pred_prob >= 0.5 else 0

                label = "Positif" if result == 1 else "Negatif"
                st.success(f"Hasil Sentimen: **{label}**")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses: {e}")
