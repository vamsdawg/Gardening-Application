import io
from datetime import datetime, timezone
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# app.py
# Streamlit "Snap-to-Mow" gardening assistant
# Usage: pip install streamlit opencv-python pillow numpy
# Run: streamlit run app.py

st.set_page_config(page_title="Snap-to-Mow", layout="wide", initial_sidebar_state="expanded")

# --- Helpers ---
def to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

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

def default_rule_engine(green_pct, last_mow_days, visible_brown_pct, plant_type, season):
    # Simple interpretable rules as fallback
    recs = []
    if green_pct > 0.55:
        if last_mow_days is None or last_mow_days >= 14:
            recs.append("Mow now — grass cover is dense and likely needs trimming.")
        else:
            recs.append("Consider mowing soon; last mow was {} days ago.".format(last_mow_days))
    elif green_pct > 0.15:
        recs.append("Hold off on mowing; grass is patchy. Consider inspecting for pests or nutrient issues.")
    else:
        recs.append("Low green cover detected — may need restoration (reseeding, soil improvement, or watering).")

    if visible_brown_pct > 0.05:
        recs.append("Brown or dry patches detected; inspect for pests, disease, or drought stress.")
    if plant_type.lower() in ("shrub", "flower", "vine"):
        recs.append("If pruning woody plants: prune in the appropriate season (usually late winter/early spring for many species).")
    if season.lower() in ("winter",):
        recs.append("Seasonal note: growth is slower in winter; avoid heavy pruning.")
    return "\n".join(recs)

def llm_integration(prompt: str, api_endpoint: str = None):
    """
    Placeholder for your custom LLM integration.
    Return a string with recommendations, or None to fall back to default_rule_engine.
    Replace/extend this function to call your LLM/service.
    """
    return None

def make_report_text(summary, metrics, meta):
    t = []
    t.append("Snap-to-Mow Report")
    # Use timezone-aware UTC
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

# --- UI ---
st.title("Snap-to-Mow — Visual, data-driven lawn & plant care")
st.write("Upload a photo of your lawn/plant. The app segments green cover and produces actionable recommendations.")

with st.sidebar:
    st.header("Options & Inputs")
    llm_endpoint = st.text_input("External LLM API endpoint (optional)", help="Provide your LLM endpoint if you want remote recommendations")
    segmentation_method = st.selectbox("Segmentation method", ["Color (fast)", "Color + tune sliders"])
    st.markdown("Provide a few context inputs to improve recommendations:")
    last_mow_days = st.number_input("Days since last mow/prune (if known)", min_value=0, max_value=365, value=14)
    plant_type = st.selectbox("Plant type", ["Lawn/Grass", "Shrub", "Flowering plant", "Mixed/Other"])
    season = st.selectbox("Season", ["spring", "summer", "autumn", "winter"])
    submit_btn = st.button("Run analysis")

# main column for image and results
col1, col2 = st.columns([1, 1])

uploaded = st.file_uploader("Upload an image (jpg/png). Try an overhead lawn shot or a close plant photo.", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    arr = np.array(image)
    st.sidebar.image(image, caption="Uploaded image preview", use_column_width=True)

    # default HSV parameters and interactive tuning if chosen
    if segmentation_method == "Color + tune sliders":
        st.sidebar.markdown("Tune HSV thresholds for 'green' (advanced)")
        lower_h = st.sidebar.slider("Lower Hue", 25, 85, 35)
        upper_h = st.sidebar.slider("Upper Hue", 60, 100, 85)
        sat_min = st.sidebar.slider("Min Saturation", 0, 255, 40)
        val_min = st.sidebar.slider("Min Value (brightness)", 0, 255, 40)
        morph_k = st.sidebar.slider("Morph kernel size", 1, 25, 7)
    else:
        lower_h, upper_h, sat_min, val_min, morph_k = 35, 85, 40, 40, 7

    if submit_btn:
        with st.spinner("Running segmentation and decision engine..."):
            mask = segment_green_cv(arr, lower_h, upper_h, sat_min, val_min, morph_k)
            overlay = overlay_mask(arr, mask, color=(0, 200, 0), alpha=0.45)
            green_frac = percent_coverage(mask)
            # crude visible brown estimate: detect low-value pixels that are not green
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            brown_mask = (mask == 0) & (gray < 150)
            brown_frac = int(100 * brown_mask.sum() / mask.size) / 100.0

            # Build prompt for LLM
            metrics = {
                "green_coverage_pct": f"{green_frac*100:.1f}%",
                "estimated_brown_pct": f"{brown_frac*100:.1f}%",
            }
            meta = {"last_mow_days": int(last_mow_days), "plant_type": plant_type, "season": season}

            prompt = (
                f"You are a practical gardening assistant. A user uploaded an image of their {plant_type}. "
                f"Computed metrics: green coverage {green_frac*100:.1f}%, brown/dry approx {brown_frac*100:.1f}%. "
                f"User inputs: last_mow_days={last_mow_days}, season={season}. "
                "Provide concise, actionable recommendations (1-3 bullets) about whether to mow, prune, wait, "
                "and suggest simple products or interventions if relevant. Mention confidence and immediate next steps."
            )

            llm_resp = None
            if llm_endpoint:
                llm_resp = llm_integration(prompt, llm_endpoint)
            if llm_resp is None:
                summary = default_rule_engine(green_frac, last_mow_days, brown_frac, plant_type, season)
                used_llm = False
            else:
                summary = llm_resp
                used_llm = True

        # Results display
        col1.image(image, caption="Original", use_column_width=True)
        col2.image(overlay, caption="Segmentation overlay", use_column_width=True)

        st.markdown("### Analysis")
        st.write("Green coverage:", f"{green_frac*100:.1f}%")
        st.write("Estimated brown/dry visible:", f"{brown_frac*100:.1f}%")
        st.markdown("### Recommendation")
        st.write(summary)
        if not used_llm:
            st.info("Using local rule-based recommendations. Provide an external LLM endpoint in the sidebar for more detailed suggestions.")

        # Offer downloads
        mask_pil = Image.fromarray(mask)
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        bmask = buf_mask.getvalue()
        st.download_button("Download mask (png)", data=bmask, file_name="mask.png", mime="image/png")

        # small report
        report = make_report_text(summary, metrics, meta)
        st.download_button("Download report (.txt)", data=report, file_name="snap_to_mow_report.txt", mime="text/plain")

        st.success("Done. Use the sliders (in sidebar) to tweak segmentation if you need better masks.")

else:
    st.info("Upload an image to begin. Try an overhead lawn or close plant photo taken in natural light.")
