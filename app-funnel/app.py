import streamlit as st
import joblib
from src.text_preprocessor import TextPreprocessor

# Cargar modelos y objetos
brand_model = joblib.load("../models/brand_classifier.pkl")
funnel_model = joblib.load("../models/funnel_classifier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")
encoder_brand = joblib.load("../models/encoder_brand.pkl")
encoder_funnel = joblib.load("../models/encoder_funnel.pkl")


preprocessor = TextPreprocessor()

st.title("🔍 Clasificador de Marca y Funnel Stage")

user_input = st.text_area("✏️ Ingresa el texto:", height=200)

if st.button("🔎 Predecir"):
    if user_input.strip() == "":
        st.warning("⚠️ Por favor, introduce un texto.")
    else:
        clean_text = preprocessor.transform(user_input)
        X = vectorizer.transform([clean_text])
        pred_brand = brand_model.predict(X)
        pred_funnel = funnel_model.predict(X)

        label_brand = encoder_brand.inverse_transform(pred_brand)[0]
        label_funnel = encoder_funnel.inverse_transform(pred_funnel)[0]

        st.subheader("📊 Resultados:")
        st.write("**Marca predicha:**", f"🛍️ {label_brand}")
        st.write("**Etapa del embudo:**", f"🪜 {label_funnel}")
