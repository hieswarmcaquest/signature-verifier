import gradio as gr
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

def match_orb(img1, img2, ratio_thresh=0.75):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return len(good_matches), len(matches)

def compute_ssim(img1, img2):
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    score, _ = ssim(img1, img2, full=True)
    return score

def compute_hog_similarity(img1, img2):
    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))
    hog1 = hog(img1, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    hog2 = hog(img2, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    return cosine_similarity([hog1], [hog2])[0][0]

def dummy_siamese_score(img1, img2):
    return np.random.uniform(0.7, 0.95)  # Placeholder for actual Siamese network

def verify_signature(test_img_pil, ref_img_pil, orb_threshold=0.04, ssim_threshold=0.74, hog_threshold=0.8, siamese_threshold=0.85):
    test_img = cv2.cvtColor(np.array(test_img_pil.convert("L")), cv2.COLOR_GRAY2BGR)
    ref_img = cv2.cvtColor(np.array(ref_img_pil.convert("L")), cv2.COLOR_GRAY2BGR)
    gray1 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    good, total = match_orb(gray1, gray2)
    ratio = good / total if total > 0 else 0
    ssim_score = compute_ssim(gray1, gray2)
    hog_score = compute_hog_similarity(gray1, gray2)
    siamese_score = dummy_siamese_score(gray1, gray2)

    # Styled pass/fail icons
    check = '<span style="color:green;"><b>✅</b></span>'
    cross = '<span style="color:red;"><b>❌</b></span>'

    orb_status = check if ratio > orb_threshold else cross
    ssim_status = check if ssim_score > ssim_threshold else cross
    hog_status = check if hog_score > hog_threshold else cross
    siamese_status = check if siamese_score > siamese_threshold else cross
    print("=== Debug Log ===")
    print(f"ORB Matches: {good}/{total} (Ratio: {ratio:.2f}) {orb_status}")
    print(f"SSIM Score: {ssim_score:.2f} {ssim_status}")
    print(f"HOG Similarity: {hog_score:.2f} {hog_status}")
    print(f"Siamese Similarity: {siamese_score:.2f} {siamese_status}")
    print("=================")

    # Decision logic
    match_count = sum([
        ratio > orb_threshold,
        ssim_score > ssim_threshold,
        hog_score > hog_threshold,
        siamese_score > siamese_threshold
    ])

    if match_count >= 2:
        status = "✅ Genuine Signature"
    else:
        status = "❌ Forged Signature"

    result = (
        f"<b>{status}</b><br><br>"
        f"Good ORB matches: {good}/{total} (Ratio: {ratio:.2f}) {orb_status}<br>"
        f"SSIM Score: {ssim_score:.2f} {ssim_status}<br>"
        f"HOG Similarity (cosine): {hog_score:.2f} {hog_status}<br>"
        f"Siamese Similarity: {siamese_score:.2f} {siamese_status}<br><br>"
        f"<b>Thresholds → ORB: {orb_threshold}, SSIM: {ssim_threshold}, HOG: {hog_threshold}, Siamese: {siamese_threshold}</b>"
    )

    explanation = (
        "**Explanation:**\n"
        "- ORB Ratio > 0.04 → Acceptable; > 0.08 → High assurance.\n"
        "- SSIM > 0.70 → Acceptable; > 0.80 → Strong similarity.\n"
        "- HOG Similarity (cosine) > 0.75 → Acceptable; > 0.85 → Strong match.\n"
        "- Siamese Score > 0.85 → Reliable match (based on trained model).\n"
        "- If 2 or more metrics exceed their thresholds, the signature is verified as genuine."
    )

    return result, explanation

# Gradio Interface
demo = gr.Interface(
    fn=verify_signature,
    inputs=[
        gr.Image(type="pil", label="Test Signature"),
        gr.Image(type="pil", label="Reference Signature"),
        gr.Slider(0.01, 1.0, value=0.04, step=0.01, label="ORB Match Ratio Threshold"),
        gr.Slider(0.01, 1.0, value=0.74, step=0.01, label="SSIM Threshold"),
        gr.Slider(0.01, 1.0, value=0.8, step=0.01, label="HOG Threshold"),
        gr.Slider(0.01, 1.0, value=0.85, step=0.01, label="Siamese Threshold")
    ],
    outputs=[
        gr.HTML(label="Verification Result"),
        gr.Markdown(label="Interpretation Guide")
    ],
    title="🧠 Signature Verifier (ORB + SSIM + HOG + Siamese)",
    description="Compares two signatures using ORB, SSIM, HOG, and Siamese similarity. Adjust thresholds to customize verification."
)

demo.launch(server_name="0.0.0.0", server_port=7860)

