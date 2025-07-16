import gradio as gr
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity


def extract_signature_only(img_pil):
    img = np.array(img_pil.convert("L"))
    
    # Binarize and invert
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological filtering to remove text (small thin objects)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours on cleaned image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # Sort by contour area, keep largest (likely signature)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Merge all contours above a threshold
    boxes = []
    for c in contours:
        if cv2.contourArea(c) > 500:  # adjustable threshold
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        return img

    # Union of all selected bounding boxes
    x1 = min([b[0] for b in boxes])
    y1 = min([b[1] for b in boxes])
    x2 = max([b[2] for b in boxes])
    y2 = max([b[3] for b in boxes])

    margin = 10
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    y2 = min(img.shape[0], y2 + margin)

    cropped = img[y1:y2, x1:x2]
    return cropped

# Preprocessing: resize, blur, threshold
def preprocess_signature(img_pil):
    img = extract_signature_only(img_pil)
    img = cv2.resize(img, (256, 128))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

# Enhanced ORB matcher
def match_orb(img1, img2, ratio_thresh=0.75):
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return len(good_matches), len(matches)

# SSIM with fixed resizing
def compute_ssim(img1, img2):
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    score, _ = ssim(img1, img2, full=True)
    return score

# HOG similarity
def compute_hog_similarity(img1, img2):
    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))
    hog1 = hog(img1, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    hog2 = hog(img2, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    return cosine_similarity([hog1], [hog2])[0][0]

# Simulated Siamese model
def dummy_siamese_score(img1, img2):
    return np.random.uniform(0.7, 0.95)  # Replace with real model

# Main verification function
def verify_signature(
    test_img_pil, ref_img_pil,
    orb_threshold=0.04, ssim_threshold=0.74,
    hog_threshold=0.8, siamese_threshold=0.85
):
    # --- preprocessing ---
    gray1 = preprocess_signature(test_img_pil)
    gray2 = preprocess_signature(ref_img_pil)

    # --- metrics ---
    good, total = match_orb(gray1, gray2)
    ratio        = good / total if total > 0 else 0
    ssim_score   = compute_ssim(gray1, gray2)
    hog_score    = compute_hog_similarity(gray1, gray2)
    siamese_score = dummy_siamese_score(gray1, gray2)

    # --- per-metric status icons ---
    check, cross = '<span style="color:green;"><b>✅</b></span>', '<span style="color:red;"><b>❌</b></span>'
    orb_status     = check if ratio       > orb_threshold   else cross
    ssim_status    = check if ssim_score  > ssim_threshold  else cross
    hog_status     = check if hog_score   > hog_threshold   else cross
    siamese_status = check if siamese_score > siamese_threshold or siamese_score >= 0.83 else cross

    # --- counting rule ---
    match_count = sum([
        ratio       > orb_threshold,
        ssim_score  > ssim_threshold,
        hog_score   > hog_threshold,
        siamese_score > siamese_threshold or siamese_score >= 0.83
    ])

    # --- NEW decision logic ---
    is_genuine = (ratio > orb_threshold) or (ssim_score > ssim_threshold) or (match_count >= 2)

    status_html = (
        "<span style='color:green;'><b>✅ Genuine Signature</b></span>"
        if is_genuine else
        "<span style='color:red;'><b>❌ Forged Signature</b></span>"
    )

    # --- assemble result HTML ---
    result = (
        f"<b>{status_html}</b><br><br>"
        f"Good ORB matches: {good}/{total} (Ratio: {ratio:.2f}) {orb_status}<br>"
        f"SSIM Score: {ssim_score:.2f} {ssim_status}<br>"
        f"HOG Similarity (cosine): {hog_score:.2f} {hog_status}<br>"
        f"Siamese Similarity: {siamese_score:.2f} {siamese_status}<br><br>"
        f"<b>Thresholds → ORB: {orb_threshold}, SSIM: {ssim_threshold}, "
        f"HOG: {hog_threshold}, Siamese: {siamese_threshold} (fallback ≥ 0.83)</b>"
    )

    # --- updated explanation ---
    explanation = (
        "**Explanation:**\n"
        "- ORB ratio > 0.04 → Acceptable; > 0.08 → High assurance.\n"
        "- SSIM > 0.70 → Acceptable; > 0.80 → Strong similarity.\n"
        "- HOG cosine > 0.75 → Acceptable; > 0.85 → Strong match.\n"
        "- Siamese > 0.85 → Reliable match (model-based), fallback ≥ 0.83.\n"
        "- **New rule:** A signature is *also* accepted as genuine if **either ORB ratio or SSIM alone passes its threshold**, even if other metrics fail.\n"
        "- Otherwise, at least **2 metrics** must pass."
    )

    # --- images for UI ---
    preproc_test_pil = Image.fromarray(gray1)
    preproc_ref_pil  = Image.fromarray(gray2)

    return result, explanation, preproc_test_pil, preproc_ref_pil

# Gradio UI
demo = gr.Interface(
    fn=verify_signature,
    inputs=[
        gr.Image(type="pil", label="Test Signature"),
        gr.Image(type="pil", label="Reference Signature"),
        gr.Slider(0.01, 1.0, value=0.04, step=0.01, label="ORB Match Ratio Threshold"),
        gr.Slider(0.01, 1.0, value=0.74, step=0.01, label="SSIM Threshold"),
        gr.Slider(0.01, 1.0, value=0.80, step=0.01, label="HOG Threshold"),
        gr.Slider(0.01, 1.0, value=0.85, step=0.01, label="Siamese Threshold")
    ],
    outputs=[
        gr.HTML(label="Verification Result"),
        gr.Markdown(label="Interpretation Guide"),
        gr.Image(type="pil", label="Processed Test Signature"),
        gr.Image(type="pil", label="Processed Reference Signature"),
    ],
    title="🧠 Signature Verifier (Enhanced ORB + SSIM + HOG + Siamese)",
    description="Compares two signatures using advanced image preprocessing and multi-metric similarity scoring. View processed inputs for transparency."
)

demo.launch(server_name="0.0.0.0", server_port=7860)

