# streamlit_app_full_advanced.py

import streamlit as st
import numpy as np
import re
import pickle
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd

# --- Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Preprocessing ---
def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z']", ' ', text).lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Load All Models ---
model_nb = joblib.load("naive_bayes_model.pkl")
model_svm = joblib.load("svm_model.pkl")
model_cnn = load_model("cnn_model.h5")

vectorizer = joblib.load("tfidf_vectorizer.pkl")

with open("cnn_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 200  # Harus sesuai training CNN

# --- Dummy Accuracy (bisa ganti ke hasil evaluasi asli) ---
accuracy = {
    "Naive Bayes": 0.85,
    "SVM": 0.88,
    "CNN": 0.90
}

# --- Inisialisasi session_state untuk histori ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Streamlit UI ---
st.set_page_config(page_title="Analisis Sentimen Film", layout="centered")
st.title("🎬 Analisis Sentimen Ulasan Film")
st.markdown("Masukkan ulasan film, pilih model, dan lihat hasil prediksi serta perbandingan performa model.")

text_input = st.text_area("Masukkan ulasan film:")
model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "SVM", "CNN"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Analisis"):
        if not text_input.strip():
            st.warning("Teks ulasan tidak boleh kosong.")
        else:
            cleaned = preprocess(text_input)

            if model_choice in ["Naive Bayes", "SVM"]:
                vectorized = vectorizer.transform([cleaned])
                result = model_nb.predict(vectorized)[0] if model_choice == "Naive Bayes" else model_svm.predict(vectorized)[0]
            else:  # CNN
                sequence = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(sequence, maxlen=MAXLEN)
                result = np.argmax(model_cnn.predict(padded), axis=1)[0]

            label = "Positif" if result == 1 else "Negatif"
            st.success(f"Hasil Sentimen: **{label}**")

            # Simpan ke histori
            st.session_state.history.append({
                "Input": text_input,
                "Model": model_choice,
                "Sentimen": label
            })

with col2:
    st.markdown("### 📊 Akurasi Model")
    for model, acc in accuracy.items():
        st.markdown(f"- **{model}**: {acc*100:.2f}%")

    # Bar chart
    df_acc = pd.DataFrame.from_dict(accuracy, orient='index', columns=['Akurasi'])
    st.bar_chart(df_acc)

# --- Riwayat ---
if st.checkbox("📝 Tampilkan Riwayat Prediksi"):
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("Belum ada riwayat prediksi.")
