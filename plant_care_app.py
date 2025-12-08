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

# Streamlit "Smart Garden Assistant" - Plant Care Analysis
# Usage: pip install -r requirements.txt
# Run: streamlit run plant_care_app.py

st.set_page_config(page_title="Smart Garden Assistant - Plant Care", layout="wide", initial_sidebar_state="expanded")

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
st.title("🌱 Plant Species Identification")

# Initialize session state for plant analysis
if 'plant_analyzed' not in st.session_state:
    st.session_state.plant_analyzed = False

# Check which identification method to use
if USE_PLANTNET and PLANTNET_API_KEY:
    st.write("🌍 Powered by PlantNet - Identifies 40,000+ plant species worldwide!")
    identification_method = "plantnet"
else:
    st.write("Upload a photo of your plant to identify the species. Detailed care recommendations coming soon!")
    st.info("💡 To unlock identification of 40,000+ plant species, add your PlantNet API key to the sidebar!")
    identification_method = "custom"

# Load custom model (as fallback or primary)
plant_model, class_indices = load_plant_model()

with st.sidebar:
    st.subheader("Plant Care Options")
    
    # API Key input
    if not PLANTNET_API_KEY:
        api_key_input = st.text_input(
            "PlantNet API Key (optional)",
            type="password",
            help="Get a free key at https://my.plantnet.org/account/doc (500 requests/day)"
        )
        if api_key_input:
            st.session_state['plantnet_key'] = api_key_input
            st.success("✅ API key set! Click 'Identify Plant' to use PlantNet.")
    else:
        st.success("✅ PlantNet API configured")
    
    plant_prompt = st.text_area(
        "Describe plant issues (optional)", 
        placeholder="e.g., Leaves turning brown, spots appearing...",
        key="plant_prompt",
        help="Your observations will be used for personalized recommendations once LLM integration is complete."
    )

# Show upload and analyze button only if not analyzed yet
if not st.session_state.plant_analyzed:
    plant_uploaded = st.file_uploader(
        "Upload plant image (jpg/png/webp)", 
        type=["jpg", "jpeg", "png", "webp"],
        key="plant_upload",
        accept_multiple_files=False
    )
    
    if plant_uploaded:
        if st.button("🔍 Identify Plant", key="plant_submit_main"):
            st.session_state.plant_analyzed = True
            st.session_state.plant_image = plant_uploaded
            st.rerun()

# Show results if analyzed
if st.session_state.plant_analyzed and 'plant_image' in st.session_state:
    image = Image.open(st.session_state.plant_image).convert("RGB")
    arr = np.array(image)
    
    # Determine which API key to use
    active_api_key = st.session_state.get('plantnet_key', PLANTNET_API_KEY)
    use_plantnet_now = USE_PLANTNET and active_api_key
    
    with st.spinner("Identifying plant..."):
        if use_plantnet_now:
            # Use PlantNet API
            result = classify_plant_plantnet(arr, active_api_key)
            
            if result['success']:
                top = result['top_result']
                plant_name = top['scientific_name']
                confidence = top['confidence']
                all_results = result['all_results']
                common_names = top['common_names']
                family = top['family']
                genus = top['genus']
                method = "PlantNet API"
            else:
                plant_name = "Unknown"
                confidence = 0.0
                all_results = []
                error_msg = result.get('message', 'Unknown error')
                method = "PlantNet API (failed)"
        else:
            # Use custom model
            if plant_model is not None:
                plant_name, confidence, predictions = classify_plant(arr, plant_model, class_indices)
                method = "Custom Model"
            else:
                plant_name, confidence = "Unknown", 0.0
                predictions = None
                method = "No model available"
        
        metrics = {
            "plant_species": plant_name if plant_name else "Unknown",
            "confidence": f"{confidence*100:.1f}%" if confidence > 0 else "N/A",
            "method": method
        }
        meta = {
            "user_notes": plant_prompt if plant_prompt else "None"
        }
        
        # Determine current season (basic approximation)
        current_month = datetime.now().month
        if current_month in [3, 4, 5]:
            season = "Spring"
        elif current_month in [6, 7, 8]:
            season = "Summer"
        elif current_month in [9, 10, 11]:
            season = "Fall"
        else:
            season = "Winter"
        
        # Pass PlantNet result data to rule engine for LLM
        plant_data = result if use_plantnet_now else None
        summary = plant_rule_engine(
            plant_name if plant_name else "Unknown",
            plant_data=plant_data,
            user_observation=plant_prompt if plant_prompt else "",
            season=season
        )
    
    # Display results
    st.image(image, caption="Uploaded Plant Image", use_container_width=True)
    
    st.markdown("### 🔍 Plant Identification")
    
    if use_plantnet_now and result['success']:
        # PlantNet results
        st.success(f"**{plant_name}**")
        
        # Common names
        if common_names:
            st.write(f"**Common names:** {', '.join(common_names[:3])}")
        
        st.write(f"**Family:** {family}")
        st.write(f"**Genus:** {genus}")
        st.write(f"**Confidence:** {confidence*100:.1f}%")
        
        # Show all results
        if len(all_results) > 1:
            st.markdown("**Alternative matches:**")
            for r in all_results[1:]:
                common = ', '.join(r['common_names'][:2]) if r['common_names'] else 'No common name'
                st.write(f"{r['rank']}. **{r['scientific_name']}** ({common}) - {r['confidence_pct']}")
        
        # API credits
        remaining = result['query_info'].get('remaining_credits', 'Unknown')
        st.caption(f"🌐 Identified via PlantNet | Remaining API calls today: {remaining}")
        
    elif use_plantnet_now and not result['success']:
        # PlantNet error
        st.error(f"❌ {result['error']}")
        st.warning(result['message'])
        
    elif plant_name and confidence > 0:
        # Custom model results
        st.success(f"**{plant_name.replace('_', ' ').title()}**")
        st.write(f"**Confidence:** {confidence*100:.1f}%")
        st.warning("⚠️ Limited to 30 plant types. For broader identification, add a PlantNet API key!")
        
        # Show top 3 predictions
        if predictions is not None:
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            st.markdown("**Top 3 Predictions:**")
            for idx in top_3_idx:
                pred_name = class_indices[str(idx)].replace('_', ' ').title()
                pred_conf = predictions[idx] * 100
                st.write(f"- {pred_name}: {pred_conf:.1f}%")
    else:
        st.warning("Plant identification unavailable")
    
    st.markdown("### 📊 Care Recommendations")
    
    if USE_LLM and GEMINI_API_KEY:
        # Display LLM-generated recommendations
        st.markdown(summary)
        
        if plant_prompt:
            st.markdown("---")
            st.caption(" Your observations were analyzed and incorporated into the recommendations above.")
    else:
        # Fallback when LLM is not configured
        st.info(summary)
        st.info("🚀 **Enable LLM Integration:** Add your Gemini API key to get:\n"
               "- Detailed watering schedules\n"
               "- Specific sunlight requirements\n"
               "- Soil recommendations\n"
               "- Pest & disease identification\n"
               "- Harvest timing guidance\n"
               "- Personalized care based on your observations")
        
        if plant_prompt:
            st.markdown("### 📝 Your Observations")
            st.write(plant_prompt)
            st.info("💡 Enable LLM integration to get personalized analysis of your observations.")
    
    # Downloads
    report = make_report_text(summary, metrics, meta, "Plant Identification")
    st.download_button("Download report", data=report, file_name="plant_report.txt", mime="text/plain")
    
    st.markdown("---")
    if st.button("🔄 Analyze Another Photo", key="plant_reset"):
        st.session_state.plant_analyzed = False
        del st.session_state.plant_image
        st.rerun()
