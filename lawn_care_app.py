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
from gemini_llm import PlantCareLLM

# Load environment variables from .env file
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Streamlit "Smart Garden Assistant" - Lawn Care Analysis
# Usage: pip install -r requirements.txt
# Run: streamlit run lawn_care_app.py

st.set_page_config(page_title="Smart Garden Assistant - Lawn Care", layout="wide", initial_sidebar_state="expanded")

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Load from environment variable
USE_LLM = True  # Set to True to enable LLM recommendations

# Validate Gemini API key
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please add it to Streamlit secrets or .env file.")

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

def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5):
    colored = image.copy()
    colored[mask > 0] = (np.array(color) * alpha + colored[mask > 0] * (1 - alpha)).astype(np.uint8)
    return colored

def percent_coverage(mask: np.ndarray):
    total = mask.size
    covered = (mask > 0).sum()
    return float(covered) / float(total) if total else 0.0

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
    
    st.markdown("**Exclude Areas (ROI):**")
    st.caption("Adjust to ignore background woods, fences, or non-lawn areas.")
    col_roi1, col_roi2 = st.columns(2)
    with col_roi1:
        roi_top = st.slider("Exclude Top %", 0, 50, 15, key="roi_top")
        roi_bottom = st.slider("Exclude Bottom %", 0, 50, 0, key="roi_bottom")
    with col_roi2:
        roi_left = st.slider("Exclude Left %", 0, 50, 0, key="roi_left")
        roi_right = st.slider("Exclude Right %", 0, 50, 0, key="roi_right")

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
        
        st.sidebar.markdown("**Tune Brightness for Dead/Bald:**")
        bald_prob_thresh = st.sidebar.slider("Bald/Soil Brightness Threshold", 0, 150, 85, help="Pixels darker than this are 'Bald'")
        dead_upper_thresh = st.sidebar.slider("Dead Grass Max Brightness", 100, 255, 165, help="Pixels brighter than this are ignored (e.g. sky/stone)")
    else:
        lower_h, upper_h, sat_min, val_min, morph_k = 35, 85, 40, 40, 7
        bald_prob_thresh = 85  # Increased to capture lighter soil
        dead_upper_thresh = 165 # Decreased to ignore bright stones/fences
    
    with st.spinner("Analyzing lawn..."):
        mask = segment_green_cv(arr, lower_h, upper_h, sat_min, val_min, morph_k)
        
        # --- Apply ROI (Region of Interest) ---
        height, width = mask.shape
        # Create ROI mask (255 = keep, 0 = ignore)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate boundaries
        t = int(height * roi_top / 100)
        b = int(height * (100 - roi_bottom) / 100)
        l = int(width * roi_left / 100)
        r = int(width * (100 - roi_right) / 100)
        
        # Set active area to 255
        if b > t and r > l:
            roi_mask[t:b, l:r] = 255
        
        # Apply ROI to green mask
        mask = cv2.bitwise_and(mask, mask, mask=roi_mask)
        
        # Calculate raw pixel counts
        green_pixels = np.count_nonzero(mask)
        
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        # Dead Grass (Light Brown): Not green, and brightness between bald_thresh and dead_upper
        # Also must be within ROI
        dead_mask = (mask == 0) & (gray >= bald_prob_thresh) & (gray < dead_upper_thresh) & (roi_mask == 255)
        dead_pixels = np.count_nonzero(dead_mask)
        
        # Bald Spots (Dark Brown/Soil): Not green, and brightness < bald_thresh
        # Also must be within ROI
        bald_mask = (mask == 0) & (gray < bald_prob_thresh) & (roi_mask == 255)
        bald_pixels = np.count_nonzero(bald_mask)
        
        # Create combined overlay
        # 1. Healthy Green Grass -> Green (0, 200, 0)
        overlay = overlay_mask(arr, mask, color=(0, 200, 0), alpha=0.4)
        # 2. Dead Grass -> Yellow (255, 255, 0)
        overlay = overlay_mask(overlay, dead_mask, color=(255, 255, 0), alpha=0.4)
        # 3. Bald Spots -> Red (255, 0, 0)
        overlay = overlay_mask(overlay, bald_mask, color=(255, 0, 0), alpha=0.4)
        
        # Normalize to lawn area only (Green + Dead + Bald)
        total_lawn_pixels = green_pixels + dead_pixels + bald_pixels
        
        if total_lawn_pixels > 0:
            green_frac = green_pixels / total_lawn_pixels
            dead_frac = dead_pixels / total_lawn_pixels
            bald_frac = bald_pixels / total_lawn_pixels
        else:
            green_frac = 0.0
            dead_frac = 0.0
            bald_frac = 0.0
        
        metrics = {
            "green_coverage_pct": f"{green_frac*100:.1f}%",
            "dead_grass_pct": f"{dead_frac*100:.1f}%",
            "bald_spots_pct": f"{bald_frac*100:.1f}%",
        }
        meta = {
            "last_mow_days": int(last_mow_days), 
            "season": lawn_season,
            "user_notes": lawn_prompt if lawn_prompt else "None"
        }
        
        summary = lawn_rule_engine(
            green_frac, 
            dead_frac,
            bald_frac,
            last_mow_days, 
            lawn_season,
            user_observation=lawn_prompt if lawn_prompt else ""
        )
    
    # Display results
    col1, col2 = st.columns([1, 1])
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(overlay, caption="Green Coverage Analysis", use_container_width=True)
    
    st.markdown("### 📊 Analysis Results")
    st.write(f"**Green coverage (Healthy):** {green_frac*100:.1f}%")
    st.write(f"**Dead Grass (Light Brown):** {dead_frac*100:.1f}%")
    st.write(f"**Bald Spots (No Growth):** {bald_frac*100:.1f}%")
    
    st.markdown("### 💡 Recommendations")
    
    if USE_LLM and GEMINI_API_KEY:
        # Display LLM-generated recommendations
        st.markdown(summary)
        
        if lawn_prompt:
            st.markdown("---")
            st.caption("💡 Your concerns were analyzed and incorporated into the recommendations above.")
    else:
        # Fallback when LLM is not configured
        st.info(summary)
        
        if lawn_prompt:
            st.markdown("### 📝 Your Notes")
            st.write(lawn_prompt)
            st.info("💡 Enable LLM integration to get personalized analysis of your lawn concerns.")
    
    # Downloads
    mask_pil = Image.fromarray(mask)
    buf_mask = io.BytesIO()
    mask_pil.save(buf_mask, format="PNG")
    st.download_button("Download mask", data=buf_mask.getvalue(), file_name="lawn_mask.png", mime="image/png")
    
    report = make_report_text(summary, metrics, meta, "Lawn Care")
    st.download_button("Download report", data=report, file_name="lawn_report.txt", mime="text/plain")
    
    st.markdown("---")
    if st.button("🔄 Analyze Another Photo", key="lawn_reset"):
        st.session_state.lawn_analyzed = False
        del st.session_state.lawn_image
        st.rerun()
