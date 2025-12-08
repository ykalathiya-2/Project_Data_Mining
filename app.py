import streamlit as st
import torch
import numpy as np
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing transformers with error handling
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    st.error("Error: 'transformers' library is not properly installed. Please run `pip install transformers`.")
    st.stop()

from lime.lime_text import LimeTextExplainer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ============================================================================
# CUSTOM CSS (Minimalist & Clean)
# ============================================================================
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fake-news {
        background-color: #ffebee;
        border: 2px solid #ef5350;
        color: #c62828;
    }
    .real-news {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
TRANSFORMER_MODEL_NAME = 'roberta-base'
MAX_LENGTH = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_model_path(filename):
    """Robustly find model paths."""
    possible_paths = [
        f'./models/{filename}',
        f'../models/{filename}',
        f'/home/ykalathiya/ds_assginement/project/models/{filename}',
        f'./{filename}'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

ROBERTA_PATH = find_model_path('roberta_best.pt')

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_transformer_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME, num_labels=2
    )
    
    if ROBERTA_PATH:
        try:
            model.load_state_dict(torch.load(ROBERTA_PATH, map_location=DEVICE))
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
    
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_transformer(text, tokenizer, model):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding='max_length')
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
    pred_idx = torch.argmax(probs, dim=1).item()
    return pred_idx, probs[0].cpu().numpy()

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📰 Fake News Detection")
    st.markdown("Analyze political statements using AI.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.info("This tool uses machine learning to estimate the probability of a statement being Fake or Real.")

    # Input
    text_input = st.text_area("Enter Statement:", height=150, placeholder="Type or paste the statement here...")

    if st.button("Analyze Statement"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        # Load Model
        with st.spinner("Loading model..."):
            tokenizer, model = load_transformer_model()

        # Predict
        with st.spinner("Analyzing..."):
            pred, probs = predict_transformer(text_input, tokenizer, model)

        # Display Results
        st.markdown("### Analysis Results")
        
        # 0 = Fake, 1 = Real (Assuming this mapping based on previous code)
        is_fake = (pred == 0)
        confidence = probs[pred]
        
        if is_fake:
            st.markdown(f"""
                <div class="result-card fake-news">
                    <h1>🚨 FAKE NEWS DETECTED</h1>
                    <h3>Confidence: {confidence*100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card real-news">
                    <h1>✅ LIKELY REAL</h1>
                    <h3>Confidence: {confidence*100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Probability Breakdown")
        
        # Custom Progress Bars
        st.write(f"**Fake Probability:** {probs[0]*100:.1f}%")
        st.progress(float(probs[0]))
        
        st.write(f"**Real Probability:** {probs[1]*100:.1f}%")
        st.progress(float(probs[1]))

        # XAI Section
        st.markdown("---")
        with st.expander("See Why (Explainability)"):
            try:
                st.write("Generating explanation...")
                explainer = LimeTextExplainer(class_names=['FAKE', 'REAL'])
                
                def predictor_func(texts):
                    # Handle batch of texts from LIME
                    cleaned_texts = [clean_text(t) for t in texts]
                    inputs = tokenizer(cleaned_texts, return_tensors="pt", max_length=MAX_LENGTH, 
                                     truncation=True, padding='max_length')
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                        return torch.softmax(outputs.logits, dim=1).cpu().numpy()

                exp = explainer.explain_instance(clean_text(text_input), predictor_func, num_features=10)
                
                # Create and display the figure
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                plt.close(fig)  # Clean up to avoid memory issues
            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
                st.info("Try with a shorter or simpler statement.")

if __name__ == "__main__":
    main()
