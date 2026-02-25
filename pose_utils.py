"""
Pose detection and curvature scoring for scoliosis posture visualizer.
Uses MediaPipe Pose for landmarks, OpenCV for drawing.
"""
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# Scaling factors for curvature score (tune for 0-100 range)
SHOULDER_TILT_SCALE = 80   # pixels diff -> score contribution
HIP_TILT_SCALE = 80
TORSO_ANGLE_SCALE = 4     # degrees -> score contribution


def get_pose_landmarks(image_bgr):
    """
    Run MediaPipe Pose on image. image_bgr is BGR (OpenCV format).
    Returns (landmarks_list, success). landmarks_list is list of (x, y) in pixel coords, or None.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose.close()

    if not results.pose_landmarks:
        return None, False

    lms = []
    for lm in results.pose_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        lms.append((x, y))
    return lms, True


def draw_landmarks_and_lines(image_bgr, landmarks):
    """
    Draw shoulders line, hips line, torso midline, and key points.
    Modifies image_bgr in place, returns a copy for display.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    color_line = (0, 200, 100)   # BGR green
    color_pt = (0, 165, 255)     # BGR orange
    thickness_line = max(2, min(w, h) // 300)
    radius = max(4, min(w, h) // 150)

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    # Shoulders line
    cv2.line(img, ls, rs, color_line, thickness_line)
    # Hips line
    cv2.line(img, lh, rh, color_line, thickness_line)

    # Midpoints
    shoulder_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    # Torso midline (shoulder mid -> hip mid)
    cv2.line(img, shoulder_mid, hip_mid, (255, 200, 0), thickness_line)

    # Landmark circles
    for pt in [ls, rs, lh, rh, shoulder_mid, hip_mid]:
        cv2.circle(img, pt, radius, color_pt, -1)
        cv2.circle(img, pt, radius, (255, 255, 255), 1)

    return img


def compute_curvature_metrics(landmarks, image_height):
    """
    Compute shoulder tilt (pixels), hip tilt (pixels), torso lean (degrees from vertical).
    Returns dict with keys: shoulder_tilt_px, hip_tilt_px, torso_lean_deg,
    shoulder_mid, hip_mid (for display).
    """
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    # Vertical difference: positive = right side lower
    shoulder_tilt_px = rs[1] - ls[1]
    hip_tilt_px = rh[1] - lh[1]

    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

    # Angle of line shoulder_mid -> hip_mid vs vertical (downward = 0°)
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    if abs(dy) < 1e-6:
        torso_lean_deg = 90.0
    else:
        # Vertical is (0, 1); angle from vertical in degrees
        angle_rad = np.arctan2(abs(dx), abs(dy))
        torso_lean_deg = np.degrees(angle_rad)

    return {
        "shoulder_tilt_px": shoulder_tilt_px,
        "hip_tilt_px": hip_tilt_px,
        "torso_lean_deg": torso_lean_deg,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }


def curvature_score(metrics):
    """
    Combine metrics into a single 0-100 curvature score (higher = more asymmetry).
    """
    st = abs(metrics["shoulder_tilt_px"])
    ht = abs(metrics["hip_tilt_px"])
    tl = metrics["torso_lean_deg"]

    # Normalize to contributions (empirical scaling)
    s_part = min(100, st * (100 / 25))   # ~25 px tilt -> 100
    h_part = min(100, ht * (100 / 25))
    t_part = min(100, tl * TORSO_ANGLE_SCALE)

    # Weighted combination (shoulders and hips often matter most)
    score = 0.35 * s_part + 0.35 * h_part + 0.30 * t_part
    return min(100.0, max(0.0, score))


def get_demo_label(score):
    """Return (label_str, short_label) for display."""
    if score <= 20:
        return "Normal range", "Normal"
    if score <= 40:
        return "Mild asymmetry", "Mild"
    return "Moderate asymmetry", "Moderate"


def get_exercise_recommendations(label_short):
    """Rule-based exercise recommendations by classification."""
    if label_short == "Normal":
        return [
            "Maintain good posture: keep shoulders back and level during daily activities.",
            "Core strengthening 2–3x per week (e.g. planks, dead bugs) to support spine health.",
            "Stretch chest and hip flexors regularly to avoid tightening from sitting.",
        ]
    if label_short == "Mild":
        return [
            "Side stretches: gentle lateral bends to the tighter side to improve symmetry.",
            "Shoulder blade squeezes: hold 5–10 seconds, 10 reps, 2x daily.",
            "Hip leveling: standing on the lower-hip side leg with slight bend; hold 30s.",
            "Core stability: bird-dog and cat-cow to support neutral spine.",
            "Consider a posture check with a physiotherapist for a tailored plan.",
        ]
    # Moderate
    return [
        "Professional assessment recommended: see a physiotherapist or spine specialist.",
        "Avoid self-correction only; follow a prescribed exercise program.",
        "Schroth-based or similar asymmetric exercises may be suggested by a specialist.",
        "Daily mobility: gentle torso rotation and side bends within comfort.",
        "Strengthen core and back extensors as advised by your provider.",
    ]
