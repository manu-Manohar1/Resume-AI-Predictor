import streamlit as st
import pickle
import numpy as np
import base64
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import PyPDF2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume AI Predictor",
    page_icon="🚀",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ---------------- THEME TOGGLE ----------------
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_gradient = """
    <style>
    .stApp {
        background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c1c1c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
    }
    @keyframes gradient {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }
    </style>
    """
else:
    bg_gradient = """
    <style>
    .stApp {
        background: linear-gradient(-45deg,#f5f7fa,#c3cfe2,#e2ebf0,#ffffff);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: black;
    }
    </style>
    """

st.markdown(bg_gradient, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>🚀 Resume Category Predictor</h1>
<p style='text-align:center;'>AI-powered resume classification system</p>
""", unsafe_allow_html=True)

# ---------------- INPUT METHOD ----------------
input_method = st.radio(
    "Choose Input Method",
    ["Upload PDF", "Paste Text"]
)

resume_text = ""

if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

else:
    resume_text = st.text_area("Paste your resume text below", height=200)

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict Category"):

    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        with st.spinner("Analyzing resume..."):
            X = vectorizer.transform([resume_text])
            probs = model.predict_proba(X)[0]

            top_indices = np.argsort(probs)[::-1][:3]

            st.success("Prediction Complete!")

            results = []

            for i in top_indices:
                category = label_encoder.inverse_transform([i])[0]
                confidence = round(probs[i] * 100, 2)

                st.write(f"### {category}")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence}%")
                st.write("---")

                results.append((category, confidence))

            # -------- PDF DOWNLOAD --------
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []

            styles = getSampleStyleSheet()
            elements.append(Paragraph("Resume Prediction Result", styles["Title"]))
            elements.append(Spacer(1, 20))

            data = [["Category", "Confidence %"]]
            for r in results:
                data.append([r[0], str(r[1])])

            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('GRID',(0,0),(-1,-1),1,colors.black)
            ]))

            elements.append(table)
            doc.build(elements)

            st.download_button(
                label="Download Result as PDF",
                data=buffer.getvalue(),
                file_name="prediction_result.pdf",
                mime="application/pdf"
            )