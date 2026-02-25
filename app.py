"""
Scoliosis Posture Visualizer — Streamlit demo app.
Tabs: Scan (upload/capture) | Results (overlay + score + exercises) | History / Compare.
"""
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from pose_utils import (
    get_pose_landmarks,
    draw_landmarks_and_lines,
    compute_curvature_metrics,
    curvature_score,
    get_demo_label,
    get_exercise_recommendations,
)
from history_utils import (
    ensure_history_dir,
    save_scan,
    list_entries,
    get_entry_image_bytes,
)

st.set_page_config(
    page_title="Scoliosis Posture Visualizer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ensure history folder exists
ensure_history_dir()

# Session state for current scan results
if "current_image_bytes" not in st.session_state:
    st.session_state.current_image_bytes = None
if "current_overlay_bytes" not in st.session_state:
    st.session_state.current_overlay_bytes = None
if "current_score" not in st.session_state:
    st.session_state.current_score = None
if "current_label" not in st.session_state:
    st.session_state.current_label = None
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = None
if "current_landmarks" not in st.session_state:
    st.session_state.current_landmarks = None
if "pose_success" not in st.session_state:
    st.session_state.pose_success = None


def bytes_to_bgr(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def bgr_to_bytes(img_bgr, format=".jpg"):
    _, buf = cv2.imencode(format, img_bgr)
    return buf.tobytes()


def run_pipeline(image_bytes):
    """Run pose detection and scoring; store results in session state."""
    img_bgr = bytes_to_bgr(image_bytes)
    if img_bgr is None:
        st.session_state.pose_success = False
        return
    landmarks, success = get_pose_landmarks(img_bgr)
    st.session_state.pose_success = success
    if not success or landmarks is None:
        return
    st.session_state.current_landmarks = landmarks
    overlay = draw_landmarks_and_lines(img_bgr, landmarks)
    st.session_state.current_overlay_bytes = bgr_to_bytes(overlay)
    metrics = compute_curvature_metrics(landmarks, img_bgr.shape[0])
    st.session_state.current_metrics = metrics
    score = curvature_score(metrics)
    st.session_state.current_score = score
    _, short_label = get_demo_label(score)
    st.session_state.current_label = short_label
    st.session_state.current_image_bytes = image_bytes


# ----- Tabs -----
tab_scan, tab_results, tab_history = st.tabs(["📷 Scan", "📊 Results", "📁 History / Compare"])

with tab_scan:
    st.header("Upload or capture a back photo")
    st.caption("For best results, use a clear back view with shoulders and hips visible.")

    col_up, col_cam = st.columns(2)
    with col_up:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_scan")
    with col_cam:
        cam_photo = st.camera_input("Or capture with camera", key="cam_scan")

    source = None
    if uploaded is not None:
        source = uploaded.getvalue()
    elif cam_photo is not None:
        source = cam_photo.getvalue()

    if source is not None:
        st.image(source, caption="Uploaded / captured image", use_container_width=True)
        if st.button("Run posture analysis", type="primary", key="run_scan"):
            with st.spinner("Detecting pose and computing score…"):
                run_pipeline(source)
            st.success("Analysis complete. Open the **Results** tab.")
            st.rerun()
    else:
        st.info("Upload an image or take a photo to start.")

with tab_results:
    st.header("Results")
    if st.session_state.current_score is not None and st.session_state.pose_success:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Processed image")
            st.image(
                st.session_state.current_overlay_bytes,
                caption="Landmarks: shoulders line (green), hips line (green), torso midline (blue)",
                use_container_width=True,
            )
        with c2:
            score = st.session_state.current_score
            _, short_label = get_demo_label(score)
            st.metric("Curvature score (demo)", f"{score:.1f} / 100")
            st.caption("0–20: Normal range · 21–40: Mild asymmetry · 41+: Moderate asymmetry")
            st.info(f"**Classification:** {short_label}")

            m = st.session_state.current_metrics
            if m:
                with st.expander("Raw metrics"):
                    st.write(f"- Shoulder tilt: {m['shoulder_tilt_px']:.1f} px")
                    st.write(f"- Hip tilt: {m['hip_tilt_px']:.1f} px")
                    st.write(f"- Torso lean: {m['torso_lean_deg']:.1f}°")

            st.subheader("Exercise recommendations")
            for rec in get_exercise_recommendations(short_label):
                st.markdown(f"- {rec}")

            if st.button("Save to history", key="save_res"):
                entry = save_scan(
                    st.session_state.current_image_bytes,
                    score,
                    short_label,
                    m or {},
                )
                st.success(f"Saved as {entry['timestamp'][:19]}")
                st.rerun()

        st.markdown("---")
        st.warning(
            "**Disclaimer:** This is a demo only — not a medical device or diagnosis. "
            "Consult a healthcare professional for any posture or spine-related concerns."
        )
    elif st.session_state.pose_success is False:
        st.error("Pose could not be detected. Try a clearer back view with good lighting.")
    else:
        st.info("Run a scan from the **Scan** tab to see results here.")

with tab_history:
    st.header("History & before–after comparison")
    entries = list_entries()
    if not entries:
        st.info("No saved scans yet. Save a result from the **Results** tab.")
    else:
        st.caption(f"Saved scans: {len(entries)}")
        ids = [e["id"] for e in entries]
        labels_display = [
            f"{e['timestamp'][:16]} — Score: {e['score']} ({e['label']})"
            for e in entries
        ]

        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox(
                "Before (or first image)",
                range(len(ids)),
                format_func=lambda i: labels_display[i],
                key="hist_a",
            )
        with col_b:
            sel_b = st.selectbox(
                "After (or second image)",
                range(len(ids)),
                format_func=lambda i: labels_display[i],
                key="hist_b",
                index=min(1, len(ids) - 1),
            )

        if sel_a is not None and sel_b is not None:
            id_a, id_b = ids[sel_a], ids[sel_b]
            img_bytes_a = get_entry_image_bytes(id_a)
            img_bytes_b = get_entry_image_bytes(id_b)
            e_a = entries[sel_a]
            e_b = entries[sel_b]

            if img_bytes_a and img_bytes_b:
                st.subheader("Side-by-side comparison")
                comp1, comp2 = st.columns(2)
                with comp1:
                    st.image(img_bytes_a, use_container_width=True)
                    st.caption(f"Score: {e_a['score']} — {e_a['label']}")
                with comp2:
                    st.image(img_bytes_b, use_container_width=True)
                    st.caption(f"Score: {e_b['score']} — {e_b['label']}")
                # Optional before/after slider: show two images in same column with slider
                st.markdown("---")
                st.caption("Compare intensity (slider shows blend between before ↔ after)")
                blend = st.slider("Before ← → After", 0.0, 1.0, 0.5, 0.01, key="blend")
                img_a = bytes_to_bgr(img_bytes_a)
                img_b = bytes_to_bgr(img_bytes_b)
                # Resize to same size if needed
                h = min(img_a.shape[0], img_b.shape[0])
                w = min(img_a.shape[1], img_b.shape[1])
                if img_a.shape[:2] != (h, w):
                    img_a = cv2.resize(img_a, (w, h))
                if img_b.shape[:2] != (h, w):
                    img_b = cv2.resize(img_b, (w, h))
                blended = cv2.addWeighted(img_a, 1 - blend, img_b, blend, 0)
                st.image(bgr_to_bytes(blended), use_container_width=True)

        st.markdown("---")
        st.warning("**Disclaimer:** Demo only — not a medical device or diagnosis.")
