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
from lawn_analysis import LawnAnalyzer

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

@st.cache_resource
def get_lawn_analyzer():
    """Initialize and cache LawnAnalyzer"""
    return LawnAnalyzer()

# --- Helper Functions ---
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

def lawn_rule_engine(green_pct, dead_pct, bald_pct, last_mow_days, season, user_observation="", advanced_analysis=None):
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
                    user_observation=user_observation,
                    advanced_analysis=advanced_analysis
                )
                if result['success']:
                    return result
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
    
    return {
        'success': True,
        'recommendations': "\n".join(recs),
        'product_name': None,
        'store': None,
        'reason': None,
        'usage_instructions': None,
        'image_url': None,
        'product_url': None
    }

def generate_lawn_care_recommendations(llm, green_coverage, dead_coverage, bald_coverage, last_mow_days, health_status, brown_status, season, user_observation, advanced_analysis=None):
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
    
    if advanced_analysis:
        prompt += "\nADVANCED COMPUTER VISION ANALYSIS:\n"
        if 'weeds' in advanced_analysis:
            prompt += f"- Weed Detection (YOLO): {advanced_analysis['weeds']}\n"
        if 'species' in advanced_analysis:
            prompt += f"- Species Identification (DCNN): {advanced_analysis['species']}\n"
        if 'segmentation' in advanced_analysis:
            prompt += f"- Semantic Segmentation: {advanced_analysis['segmentation']}\n"

    if user_observation:
        prompt += f"\nUSER'S CONCERN:\n{user_observation}\n"
    
    prompt += """
PROVIDE BRIEF LAWN CARE RECOMMENDATIONS using this specific Markdown format:

1. **🌿 Lawn Health Summary**
   - (Provide a 1-2 sentence summary here)

2. **✂️ Mowing Advice**
   - **Should I mow now?**
     - (Yes/No and explanation)
   - **Recommended mowing height:**
     - (Specific height advice)

3. **💧 Watering**
   - **Frequency and amount:**
     - (Advice)
   - **Best time of day:**
     - (Advice)

4. **🌱 Fertilization & Treatment**
   - **Fertilizer recommendation:**
     - (Type and timing)
   - **Treatments needed:**
     - (Advice)

5. **⚠️ Problem Areas**
   - **Cause of brown patches:**
     - (Diagnosis)
   - **Quick fix steps:**
     - (Step 1)
     - (Step 2)
"""
    
    if user_observation:
        prompt += f"""
6. **🔧 Your Concern: "{user_observation}"**
   - **Diagnosis:**
     - (Diagnosis)
   - **Action steps:**
     - (Steps)
"""
    else:
        prompt += """
6. **💡 Quick Tips**
   - (Tip 1)
   - (Tip 2)
   - (Tip 3)
"""
    
    prompt += """
CRITICAL: You must ONLY recommend products that are available at Lowe's or Home Depot.
CRITICAL: Make sure the product recommended is able to be purchased. 
CRITICAL: Make sure the product is specifically suited for lawn care based on the analysis above.

IMPORTANT: Output your response in valid JSON format containing these keys:
1. "care_guide": A markdown string containing the numbered sections 1-6 above.
2. "product_name": The specific name of the recommended product (e.g., "Scotts Turf Builder").
3. "store": Either "Lowe's" or "Home Depot".
4. "reason": A brief explanation of why this product is recommended.
5. "usage_instructions": Detailed, step-by-step application instructions. Include preparation (e.g., mow first?), application method (spreader settings if applicable), watering requirements after application, and safety precautions.
6. "image_url": A valid, publicly accessible URL to an image of the specific product. Prefer high-quality images from the manufacturer (e.g., scotts.com) or major retailers.
7. "product_url": A direct, valid URL to the specific product page on lowes.com or homedepot.com. Do NOT use a search URL. Ensure the link points to the actual item (e.g. https://www.homedepot.com/p/...).

Example JSON format:
{
  "care_guide": "1. **Summary**...",
  "product_name": "Scotts Turf Builder",
  "store": "Lowe's",
  "reason": "Contains the right mix of...",
  "usage_instructions": "1. Apply to dry lawn... 2. Water immediately...",
  "image_url": "https://...",
  "product_url": "https://www.lowes.com/pd/Scotts-Turf-Builder-..."
}
"""
    
    try:
        response = llm.model.generate_content(prompt)
        text = response.text.strip()
        # Clean up markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        return {
            'success': True,
            'recommendations': data.get('care_guide', 'No guide generated.'),
            'product_name': data.get('product_name'),
            'store': data.get('store'),
            'item_number': data.get('item_number'),
            'reason': data.get('reason'),
            'usage_instructions': data.get('usage_instructions'),
            'image_url': data.get('image_url'),
            'product_url': data.get('product_url')
        }
    except Exception as e:
        # Fallback: try to return raw text if JSON parsing fails but we got something
        if 'response' in locals() and response.text:
             return {
                'success': True,
                'recommendations': response.text,
                'product_name': None,
                'store': None,
                'reason': None,
                'usage_instructions': None,
                'image_url': None,
                'product_url': None
            }
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
    
    if isinstance(summary, dict):
        t.append(summary.get('recommendations', ''))
        if summary.get('product_name'):
            t.append("\nProduct Recommendation:")
            t.append(f"Product: {summary['product_name']}")
            t.append(f"Store: {summary['store']}")
            t.append(f"Reason: {summary['reason']}")
            if summary.get('usage_instructions'):
                t.append(f"\nUsage Instructions:\n{summary['usage_instructions']}")
    else:
        t.append(str(summary))
        
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
        st.subheader("Lawn Care Options:")
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
        
        with st.spinner("Analyzing lawn..."):
            analyzer = get_lawn_analyzer()
            
            # 1. Color/Texture Analysis (HSV)
            analysis_results = analyzer.analyze_health(arr)
            
            green_frac = analysis_results['green_coverage_pct']
            dead_frac = analysis_results['dead_grass_pct']
            bald_frac = analysis_results['bald_spots_pct']
            mask = analysis_results['masks']['green']
            
            # 2. Object Detection (YOLO) - Placeholder
            weed_results = analyzer.detect_weeds_yolo(arr)
            
            # 3. Species Identification (DCNN) - Placeholder
            species_results = analyzer.identify_species_dcnn(arr)
            
            # Create combined overlay
            overlay = analyzer.overlay_masks(arr, analysis_results['masks'])

            
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
            
            # Prepare advanced analysis data for LLM
            advanced_analysis = {
                'weeds': weed_results,
                'species': species_results
            }
            
            summary = lawn_rule_engine(
                green_frac, 
                dead_frac,
                bald_frac,
                last_mow_days, 
                lawn_season,
                user_observation=lawn_prompt if lawn_prompt else "",
                advanced_analysis=advanced_analysis
            )
        
        # Display results
        col1, col2 = st.columns([1, 1])
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(overlay, caption="Green Coverage Analysis", use_container_width=True)
        
        st.markdown("### 📊 Analysis Results")
        st.write(f"**Green coverage (Healthy):** {green_frac*100:.1f}%")
        st.write(f"**Dead Grass (Light Brown):** {dead_frac*100:.1f}%")
        st.write(f"**Bald Spots (No Growth):** {bald_frac*100:.1f}%")
        
        if USE_LLM and GEMINI_API_KEY:
            # Display LLM-generated recommendations
            if isinstance(summary, dict) and 'recommendations' in summary:
                rec_col, prod_col = st.columns([2, 1])
                with rec_col:
                    st.markdown("### 💡 Recommendations")
                    st.markdown(summary['recommendations'])
                with prod_col:
                    if summary.get('product_name'):
                        st.markdown("<h3 style='text-align: center;'>🧹 Recommended Product</h3>", unsafe_allow_html=True)
                        
                        # Determine store logo and search URL
                        store = summary.get('store', 'Lowe\'s')
                        product_name = summary.get('product_name', '')
                        item_number = summary.get('item_number')
                        
                        # Normalize store name for logic
                        is_home_depot = "Home Depot" in store
                        
                        # Generate Search URL
                        clean_name = product_name.replace(' ', '%20')
                        if is_home_depot:
                            search_url = f"https://www.homedepot.com/s/{clean_name}"
                        else:
                            search_url = f"https://www.lowes.com/search?searchTerm={clean_name}"
                        
                        # Product Name (Centered, Bold, Title Font)
                        st.markdown(f"<h3 style='text-align: center; font-weight: bold;'>{product_name}</h3>", unsafe_allow_html=True)

                        # Description (Normal writing)
                        st.write(summary.get('reason', ''))
                        
                        # How to use
                        if summary.get('usage_instructions'):
                            with st.expander("📋 How to use (Click to expand)"):
                                st.write(summary['usage_instructions'])
                        
                        # Buy at: Button
                        if is_home_depot:
                            store_name = "Home Depot"
                            btn_color = "#F96302"
                        else:
                            store_name = "Lowe's"
                            btn_color = "#004990"
                        
                        st.markdown(f"""
                            <a href="{search_url}" target="_blank" style="
                                display: block;
                                width: 100%;
                                padding: 10px;
                                background-color: {btn_color};
                                color: #FFFFFF !important;
                                text-align: center;
                                text-decoration: none;
                                border-radius: 8px;
                                font-weight: bold;
                                margin-top: 5px;
                            ">
                                <span style="color: #FFFFFF !important;">🛍️ Buy at {store_name}</span>
                            </a>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("### 💡 Recommendations")
                st.markdown(summary)
            
            if lawn_prompt:
                st.markdown("---")
                st.caption("💡 Your concerns were analyzed and incorporated into the recommendations above.")
        else:
            # Fallback when LLM is not configured
            st.markdown("### 💡 Recommendations")
            if isinstance(summary, dict):
                st.info(summary['recommendations'])
            else:
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

elif page == "Plant Care":
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
