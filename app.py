import io
import os
from datetime import datetime, timezone
import json
import sys
import importlib
from dotenv import load_dotenv

# ---------- third-party imports ----------
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

# ---------- your local/project modules (no pip) ----------
from plantnet_api import PlantNetAPI
from gemini_llm import PlantCareLLM

# Load environment variables from .env file
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Streamlit "Smart Garden Assistant" - Split UI for Lawn Care and Plant Care
# Usage: pip install -r requirements.txt
# Run: streamlit run app.py

st.set_page_config(page_title="Smart Garden Assistant", layout="wide", initial_sidebar_state="expanded")

# Configuration
PLANTNET_API_KEY = '2b10wf7hRFqr7zDzwHEVO7jcu'  # PlantNet API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Load from environment variable
USE_PLANTNET = True  # Set to True to use PlantNet, False to use custom model
USE_LLM = True  # Set to True to enable LLM recommendations

# Validate Gemini API key
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please add it to Streamlit secrets or .env file.")

# --- Model Loading ---
@st.cache_resource
def load_plant_model():
    """Load the trained plant classification model"""
    try:
        model_path = 'models/plant_classifier.h5'
        class_indices_path = 'models/class_indices.json'
        
        if os.path.exists(model_path) and os.path.exists(class_indices_path):
            model = tf.keras.models.load_model(model_path)
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            return model, class_indices
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource
def get_plantnet_client():
    """Initialize and cache PlantNet API client"""
    return PlantNetAPI(PLANTNET_API_KEY)

@st.cache_resource
def get_llm_client():
    """Initialize and cache Gemini LLM client"""
    if not GEMINI_API_KEY:
        return None
    try:
        return PlantCareLLM(GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini LLM: {str(e)}")
        return None

# --- Helper Functions ---
def segment_green_cv(image: np.ndarray, lower_h=35, upper_h=85, sat_min=40, val_min=40, morph_k=7):
    """Quick HSV-based green segmentation; returns binary mask (uint8 0/255)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([lower_h, sat_min, val_min], dtype=np.uint8)
    upper = np.array([upper_h, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def segment_brown_yellow_cv(image: np.ndarray, morph_k=5):
    """Segment dead grass (yellow) and soil (brown) using HSV to exclude background."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Dead Grass (Yellow/Straw): Hue 15-35, Sat > 20, Val > 40
    # Saturation > 20 helps exclude grey fences/concrete
    lower_yellow = np.array([15, 25, 40], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Soil (Brown): Hue 0-15 (and 165-180), Sat > 25, Val 20-200
    lower_brown1 = np.array([0, 25, 20], dtype=np.uint8)
    upper_brown1 = np.array([15, 255, 200], dtype=np.uint8)
    mask_brown1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    
    lower_brown2 = np.array([165, 25, 20], dtype=np.uint8)
    upper_brown2 = np.array([180, 255, 200], dtype=np.uint8)
    mask_brown2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    
    mask_brown = cv2.bitwise_or(mask_brown1, mask_brown2)
    
    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)
    
    return mask_yellow, mask_brown

def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5):
    colored = image.copy()
    colored[mask > 0] = (np.array(color) * alpha + colored[mask > 0] * (1 - alpha)).astype(np.uint8)
    return colored

def percent_coverage(mask: np.ndarray):
    total = mask.size
    covered = (mask > 0).sum()
    return float(covered) / float(total) if total else 0.0

def classify_plant(image: np.ndarray, model, class_indices):
    """Classify the plant in the image using custom model"""
    try:
        # Preprocess image
        img_resized = cv2.resize(image, (224, 224))
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        plant_name = class_indices[str(class_idx)]
        return plant_name, confidence, predictions[0]
    except Exception as e:
        return None, 0.0, None

def classify_plant_plantnet(image: np.ndarray, api_key):
    """Classify plant using PlantNet API"""
    try:
        api = PlantNetAPI(api_key)
        result = api.identify_from_array(image, num_results=5)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'PlantNet API error: {str(e)}'
        }

def lawn_rule_engine(green_pct, dead_pct, bald_pct, last_mow_days, season, user_observation=""):
    """Generate lawn care recommendations using LLM or fallback to rules"""
    
    # Build basic analysis
    if green_pct > 0.55:
        health_status = "Healthy and dense"
    elif green_pct > 0.15:
        health_status = "Patchy with moderate coverage"
    else:
        health_status = "Poor with low green coverage"
    
    total_brown = dead_pct + bald_pct
    if total_brown > 0.05:
        brown_status = f"Significant issues detected (Dead: {dead_pct*100:.1f}%, Bald: {bald_pct*100:.1f}%)"
    else:
        brown_status = "Minimal browning or bald spots"
    
    # If LLM is enabled, use it
    if USE_LLM and GEMINI_API_KEY:
        try:
            llm = get_llm_client()
            if llm is not None:
                # Generate LLM recommendations
                result = generate_lawn_care_recommendations(
                    llm=llm,
                    green_coverage=green_pct,
                    dead_coverage=dead_pct,
                    bald_coverage=bald_pct,
                    last_mow_days=last_mow_days,
                    health_status=health_status,
                    brown_status=brown_status,
                    season=season,
                    user_observation=user_observation
                )
                if result['success']:
                    return result['recommendations']
        except Exception as e:
            # Fall back to rules if LLM fails
            pass
    
    # Fallback: Rule-based recommendations
    recs = []
    if green_pct > 0.55:
        if last_mow_days is None or last_mow_days >= 14:
            recs.append("🌿 Mow now — grass cover is dense and likely needs trimming.")
        else:
            recs.append(f"🌿 Consider mowing soon; last mow was {last_mow_days} days ago.")
    elif green_pct > 0.15:
        recs.append("⚠️ Hold off on mowing; grass is patchy. Consider inspecting for pests or nutrient issues.")
    else:
        recs.append("🔴 Low green cover detected — may need restoration (reseeding, soil improvement, or watering).")

    if dead_pct > 0.05:
        recs.append("🟤 Dead grass detected; inspect for pests, disease, or drought stress.")
    
    if bald_pct > 0.05:
        recs.append("🟤 Bald spots detected; consider reseeding or soil improvement.")
    
    if season.lower() in ("winter",):
        recs.append("❄️ Seasonal note: growth is slower in winter; avoid heavy mowing.")
    
    return "\n".join(recs)

def generate_lawn_care_recommendations(llm, green_coverage, dead_coverage, bald_coverage, last_mow_days, health_status, brown_status, season, user_observation):
    """Generate lawn care recommendations using Gemini LLM"""
    
    # Build the prompt
    prompt = f"""You are an expert lawn care specialist, turfgrass scientist, and landscape management professional. Your role is to provide highly accurate, region-appropriate, and concise lawn care recommendations based on the identified turf type, visible conditions, and symptoms.

LAWN ANALYSIS:
- Healthy Grass: {green_coverage*100:.1f}%
- Dead/Diseased Grass: {dead_coverage*100:.1f}%
- Bald Spots (Soil/No Growth): {bald_coverage*100:.1f}%
- Overall Health: {health_status}
- Brown Patch Status: {brown_status}
- Last Mow: {last_mow_days if last_mow_days else 'Unknown'} days ago
- Current Season: {season}
"""
    
    if user_observation:
        prompt += f"\nUSER'S CONCERN:\n{user_observation}\n"
    
    prompt += """
PROVIDE BRIEF LAWN CARE RECOMMENDATIONS:

1. **🌿 Lawn Health Summary** (1-2 sentences)

2. **✂️ Mowing Advice**
   - Should I mow now? (yes/no and why)
   - Recommended mowing height

3. **💧 Watering**
   - Frequency and amount
   - Best time of day

4. **🌱 Fertilization & Treatment**
   - Fertilizer recommendation (type and timing)
   - Any treatments needed?

5. **⚠️ Problem Areas**
   - What's causing brown patches?
   - Quick fix steps
"""
    
    if user_observation:
        prompt += f"""
6. **🔧 Your Concern: "{user_observation}"**
   - Diagnosis
   - Action steps
"""
    else:
        prompt += """
6. **💡 Quick Tips**
   - 3 key actions for this season
"""
    
    prompt += "\nKeep it SHORT and practical. Use bullet points. Focus on immediate actions."
    
    try:
        response = llm.model.generate_content(prompt)
        return {
            'success': True,
            'recommendations': response.text
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def plant_rule_engine(plant_name, plant_data=None, user_observation="", season="Spring"):
    """Generate plant care recommendations using LLM or fallback"""
    if not USE_LLM or not GEMINI_API_KEY:
        return f"Plant identified as: **{plant_name.replace('_', ' ').title()}**\n\nDetailed care recommendations will be available with LLM integration."
    
    try:
        llm = get_llm_client()
        if llm is None:
            return f"Plant identified as: **{plant_name.replace('_', ' ').title()}**\n\n⚠️ LLM service unavailable. Please check your API key configuration."
        
        # Extract plant data from PlantNet results or use defaults
        if plant_data and plant_data.get('success'):
            # PlantNet returns nested structure with 'top_result'
            top_result = plant_data.get('top_result', {})
            scientific_name = top_result.get('scientific_name', plant_name)
            common_names = top_result.get('common_names', [plant_name])
            family = top_result.get('family', 'Unknown')
            genus = top_result.get('genus', 'Unknown')
            confidence = top_result.get('confidence', 0.0)
        else:
            scientific_name = plant_name
            common_names = [plant_name.replace('_', ' ').title()]
            family = 'Unknown'
            genus = 'Unknown'
            confidence = 0.0
        
        # Generate recommendations using Gemini
        result = llm.generate_plant_care_recommendations(
            plant_scientific_name=scientific_name,
            plant_common_names=common_names,
            plant_family=family,
            plant_genus=genus,
            user_observation=user_observation,
            season=season,
            confidence=confidence
        )
        
        if result['success']:
            return result['recommendations']
        else:
            return f"Plant: **{scientific_name}**\n\n⚠️ Error generating recommendations: {result.get('error', 'Unknown error')}"
    
    except Exception as e:
        return f"Plant: **{plant_name}**\n\n⚠️ Error generating recommendations: {str(e)}"

def make_report_text(summary, metrics, meta, analysis_type):
    t = []
    t.append(f"Smart Garden Assistant Report - {analysis_type}")
    t.append("Generated: " + datetime.now(timezone.utc).isoformat())
    t.append("")
    t.append("Summary:")
    t.append(summary)
    t.append("")
    t.append("Metrics:")
    for k, v in metrics.items():
        t.append(f"- {k}: {v}")
    t.append("")
    t.append("Inputs:")
    for k, v in meta.items():
        t.append(f"- {k}: {v}")
    return "\n".join(t)

# --- Main UI ---
st.sidebar.title("Navigation")

def reset_page_state():
    """Reset analysis state when switching pages"""
    st.session_state.lawn_analyzed = False
    st.session_state.plant_analyzed = False
    if 'lawn_image' in st.session_state:
        del st.session_state.lawn_image
    if 'plant_image' in st.session_state:
        del st.session_state.plant_image

page = st.sidebar.radio("Go to", ["Lawn Care", "Plant Care"], on_change=reset_page_state)

if page == "Lawn Care":
    st.title("🏡 Lawn Care Analysis")
    st.write("Upload a photo of your lawn for health analysis and mowing recommendations.")
    
    # Initialize session state for lawn analysis
    if 'lawn_analyzed' not in st.session_state:
        st.session_state.lawn_analyzed = False

    with st.sidebar:
        st.subheader("Lawn Care Options")
        lawn_segmentation_method = st.selectbox(
            "Segmentation method", 
            ["Color (fast)", "Color + tune sliders"],
            key="lawn_seg"
        )
        st.markdown("**Lawn Context:**")
        last_mow_days = st.number_input("Days since last mow", min_value=0, max_value=365, value=14, key="lawn_mow")
        lawn_season = st.selectbox("Season", ["spring", "summer", "autumn", "winter"], key="lawn_season")
        lawn_prompt = st.text_area(
            "Describe your lawn challenges or goals (optional)", 
            placeholder="e.g., Brown patches appearing, want thicker grass...",
            key="lawn_prompt"
        )
    
    # Show upload and analyze button only if not analyzed yet
    if not st.session_state.lawn_analyzed:
        lawn_uploaded = st.file_uploader(
            "Upload lawn image (jpg/png/webp)", 
            type=["jpg", "jpeg", "png", "webp"],
            key="lawn_upload",
            accept_multiple_files=False
        )
        
        if lawn_uploaded:
            # Show analyze button only when image is uploaded
            if st.button("🔍 Analyze Lawn", key="lawn_submit_main"):
                st.session_state.lawn_analyzed = True
                st.session_state.lawn_image = lawn_uploaded
                st.rerun()
    
    # Show results if analyzed
    if st.session_state.lawn_analyzed and 'lawn_image' in st.session_state:
        image = Image.open(st.session_state.lawn_image).convert("RGB")
        arr = np.array(image)
        
        # Segmentation parameters
        if lawn_segmentation_method == "Color + tune sliders":
            st.sidebar.markdown("**Tune HSV thresholds for 'green':**")
            lower_h = st.sidebar.slider("Lower Hue", 25, 85, 35, key="lawn_lh")
            upper_h = st.sidebar.slider("Upper Hue", 60, 100, 85, key="lawn_uh")
            sat_min = st.sidebar.slider("Min Saturation", 0, 255, 40, key="lawn_sat")
            val_min = st.sidebar.slider("Min Value", 0, 255, 40, key="lawn_val")
            morph_k = st.sidebar.slider("Morph kernel size", 1, 25, 7, key="lawn_morph")
            
            st.sidebar.info("Dead/Bald detection now uses color (Yellow/Brown) to exclude fences/pots.")
        else:
            lower_h, upper_h, sat_min, val_min, morph_k = 35, 85, 40, 40, 7
        
        with st.spinner("Analyzing lawn..."):
            # 1. Green Mask
            mask_green = segment_green_cv(arr, lower_h, upper_h, sat_min, val_min, morph_k)
