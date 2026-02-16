import streamlit as st
import pickle
import os
import sys
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Add src to path
from preprocess import clean_text

# Page Config
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🏸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #4F46E5;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: #6B7280;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    .stTextArea textarea {
        background-color: white;
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    
    .stTextArea textarea:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    .stButton button {
        background: linear-gradient(90deg, #4F46E5 0%, #4338ca 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.3);
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .positive-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .negative-card {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Main App Structure
st.markdown('<div class="main-header"><h1>🏸 Review Sentiment Pro</h1><p>Analyze Yonex Mavis 350 reviews with AI</p></div>', unsafe_allow_html=True)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `src/train.py` first.")
else:
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        user_input = st.text_area("", placeholder="Paste a customer review here to analyze sentiment...", height=150)
        
        if st.button("✨ Analyze Sentiment"):
            if user_input.strip():
                # Preprocess
                cleaned = clean_text(user_input)
                
                # Predict
                prediction = model.predict([cleaned])[0]
                probability = model.predict_proba([cleaned])[0]
                
                # Display Result
                if prediction == 1:
                    conf = probability[1] * 100
                    st.markdown(f"""
                    <div class="result-card positive-card">
                        <h2 style="margin:0; color:white;">😊 Positive Feedback</h2>
                        <p style="margin:0.5rem 0 0 0; color:white; opacity:0.9;">Confidence: <strong>{conf:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    conf = probability[0] * 100
                    st.markdown(f"""
                    <div class="result-card negative-card">
                        <h2 style="margin:0; color:white;">😞 Negative Feedback</h2>
                        <p style="margin:0.5rem 0 0 0; color:white; opacity:0.9;">Confidence: <strong>{conf:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")

st.markdown('<div class="footer">Powered by Scikit-learn & Streamlit • Built for Flipkart Product Analysis</div>', unsafe_allow_html=True)
