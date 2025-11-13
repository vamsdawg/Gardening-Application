# =====================================================
# AI Gardening Assistant (Clean Professional Version)
# =====================================================

import io
import streamlit as st
from datetime import datetime, timezone
from PIL import Image
import numpy as np
import cv2

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="AI Gardening Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SIDEBAR UI
# =====================================================
def build_sidebar():
    with st.sidebar:
        st.header("Analysis Type")
        analysis_type = st.selectbox(
            "What do you want to analyze?",
            ["Grass Health (Mow or Not)", "Plant Health (Leaves / Flowers)"]
        )

        st.header("User Inputs")
        last_mow_days = st.number_input("Days since last mow / prune", 0, 365, 14)
        season = st.selectbox("Season", ["spring", "summer", "autumn", "winter"])

        st.markdown("---")
        st.subheader("Advanced Tuning")
        enable_sliders = st.checkbox("Enable segmentation tuning", False)

        if enable_sliders:
            lower_h = st.slider("Lower Hue", 10, 90, 35)
            upper_h = st.slider("Upper Hue", 50, 120, 85)
            sat_min  = st.slider("Min Saturation",   0, 255, 40)
            val_min  = st.slider("Min Brightness",   0, 255, 40)
            morph_k  = st.slider("Morph Kernel Size", 1, 25, 7)
        else:
            lower_h, upper_h, sat_min, val_min, morph_k = 35, 85, 40, 40, 7

        run_btn = st.button("Run analysis")

    return analysis_type, last_mow_days, season, run_btn, (lower_h, upper_h, sat_min, val_min, morph_k)


# =====================================================
# IMAGE PROCESSING HELPERS
# =====================================================
def segment_green(img, params):
    lower_h, upper_h, sat_min, val_min, morph_k = params
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([lower_h, sat_min, val_min], np.uint8)
    upper = np.array([upper_h, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def overlay_mask(img, mask):
    overlay = img.copy()
    overlay[mask > 0] = (
        0.45 * overlay[mask > 0] +
        0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)
    return overlay


def percent(mask):
    return (mask > 0).sum() / mask.size


# =====================================================
# GRASS ANALYSIS ENGINE
# =====================================================
def smart_grass_recommendation(green_pct, brown_pct, texture_score, last_mow, season):
    msg = []

    if green_pct > 75 and brown_pct < 10:
        msg.append("🌿 Lawn is very healthy and dense. Ideal for mowing.")
        if last_mow > 10:
            msg.append("⏱️ It's been a while — mowing now will keep it even.")
        return "\n".join(msg)

    if green_pct > 55:
        msg.append("👍 Lawn is mostly healthy.")
        if brown_pct > 15:
            msg.append("⚠️ Some dry patches — increase watering before mowing.")
        if texture_score < 0.3:
            msg.append("⚠️ Patchiness detected — avoid mowing until growth evens out.")
        return "\n".join(msg)

    if 20 < green_pct < 55:
        msg.append("⚠️ Lawn is patchy and uneven.")
        msg.append("💧 Increase watering and apply nitrogen fertilizer.")
        msg.append("⛔ Avoid mowing until grass fills in more.")
        return "\n".join(msg)

    msg.append("❌ Very low grass coverage.")
    msg.append("🌱 Consider reseeding or soil improvement.")
    return "\n".join(msg)


# =====================================================
# PLANT HEALTH ANALYSIS (Organzied)
# =====================================================
def analyze_plant_health(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    R, G, B = np.mean(image, axis=(0,1))

    issues = []

    # -----------------------------------------------------
    # 1. Strong brown detection (real necrosis only)
    # -----------------------------------------------------
    # Brown should be DARK + LOW saturation
    brown_pixels = ((R < 120) & (G < 110) & (B < 100))
    if np.mean(brown_pixels) > 0.10:  # at least 10% brown
        issues.append("🟤 Brown necrotic areas detected — possible disease.")
    
    # -----------------------------------------------------
    # 2. Yellowing — more strict, avoids false positives
    # -----------------------------------------------------
    if (G < 100) and (R > 120) and (S < 120):
        issues.append("🌕 Yellowing detected — nutrient deficiency possible.")

    # -----------------------------------------------------
    # 3. Rust hue — only detect if MANY pixels are orange
    # -----------------------------------------------------
    hue_mask = (H > 10) & (H < 25)
    if np.mean(hue_mask) > 0.12:
        issues.append("🟠 Rust-like orange areas — possible rust fungus.")

    # -----------------------------------------------------
    # 4. Spot detection — more strict (only dark circular spots)
    # -----------------------------------------------------
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 4
    )

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circular_spots = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 2000:
            circular_spots += 1

    if circular_spots >= 4:
        issues.append("🟤 Dark circular spots detected — fungal leaf spot likely.")

    # -----------------------------------------------------
    # 5. Leaf holes (pests)
    # -----------------------------------------------------
    holes = gray > 240
    if np.mean(holes) > 0.008:
        issues.append("🐛 Insect bite holes detected.")

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------
    if not issues:
        return "✅ Plant appears healthy — no significant issues detected."

    return "\n".join(issues)


# =====================================================
# MAIN APP LOGIC
# =====================================================
analysis_type, last_mow_days, season, run_btn, params = build_sidebar()

st.title("🌱 AI Gardening Assistant — Lawn + Plant Health Scanner")

uploaded = st.file_uploader("Upload an image (grass or plant leaf).",
                            type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    arr = np.array(img)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded Image", use_container_width=True)

    if run_btn:

        # ----------------------------------------------------------
        # GRASS ANALYSIS
        # ----------------------------------------------------------
        if analysis_type == "Grass Health (Mow or Not)":

            mask = segment_green(arr, params)
            overlay = overlay_mask(arr, mask)

            col2.image(overlay, caption="Detected Green Areas",
                       use_container_width=True)

            green_pct = percent(mask) * 100

            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            brown_mask = (mask == 0) & (gray < 145)
            brown_pct  = (brown_mask.sum() / mask.size) * 100

            texture_score = np.std(gray) / 255

            st.subheader("🌿 Lawn Analysis")
            st.write(f"Green coverage: **{green_pct:.1f}%**")
            st.write(f"Brown/dry areas: **{brown_pct:.1f}%**")

            st.subheader("Recommendation")
            st.info(
                smart_grass_recommendation(
                    green_pct, brown_pct, texture_score,
                    last_mow_days, season
                )
            )

        # ----------------------------------------------------------
        # PLANT ANALYSIS
        # ----------------------------------------------------------
        else:
            st.subheader("🩺 Plant Health Analysis")
            st.info(analyze_plant_health(arr))

else:
    st.info("Upload an image to begin.")
