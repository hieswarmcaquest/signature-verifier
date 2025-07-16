import gradio as gr
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

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

def verify_signature(test_img_pil, ref_img_pil, orb_threshold=0.04, ssim_threshold=0.74):
    test_img = cv2.cvtColor(np.array(test_img_pil.convert("L")), cv2.COLOR_GRAY2BGR)
    ref_img = cv2.cvtColor(np.array(ref_img_pil.convert("L")), cv2.COLOR_GRAY2BGR)
    gray1 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    good, total = match_orb(gray1, gray2)
    ratio = good / total if total > 0 else 0
    ssim_score = compute_ssim(gray1, gray2)

    print("=== Debug Log ===")
    print(f"ORB Matches: {good}/{total} (Ratio: {ratio:.2f})")
    print(f"SSIM Score: {ssim_score:.2f}")
    print(f"ORB Threshold Used: {orb_threshold}")
    print(f"SSIM Threshold Used: {ssim_threshold}")
    print("=================")

    if ratio > orb_threshold or ssim_score > ssim_threshold:
        status = "✅ Genuine Signature"
    else:
        status = "❌ Forged Signature"

    # Display each result line separately in bold
   # result = (
    #    f"**{status}**\n\n"
    #    f"**Good ORB matches: {good}/{total} (Ratio: {ratio:.2f})**\n"
    #    f"**SSIM Score: {ssim_score:.2f}**\n"
    #    f"**Thresholds Used → ORB: {orb_threshold}, SSIM: {ssim_threshold}**"
    #)

    result = (
        f"**{status}**<br><br>"
        f"**Good ORB matches: {good}/{total} (Ratio: {ratio:.2f})**<br>"
        f"**SSIM Score: {ssim_score:.2f}**<br>"
        f"**Thresholds Used → ORB: {orb_threshold}, SSIM: {ssim_threshold}**"
    )

    explanation = (
        "**Explanation:**\n"
        "- **ORB Match Ratio** reflects how many keypoints matched between the two signatures.\n"
        "- **SSIM (Structural Similarity)** measures image-level similarity (brightness, shape, contrast).\n"
        "- If either ORB match ratio or SSIM score exceeds the threshold, the signature is accepted as genuine.\n"
        "- Adjust thresholds using sliders to tune the sensitivity of the system."
    )

    return result, explanation

# Gradio UI
demo = gr.Interface(
    fn=verify_signature,
    inputs=[
        gr.Image(type="pil", label="Test Signature"),
        gr.Image(type="pil", label="Reference Signature"),
        gr.Slider(0.01, 1.0, value=0.04, step=0.01, label="ORB Match Ratio Threshold"),
        gr.Slider(0.01, 1.0, value=0.74, step=0.01, label="SSIM Threshold")
    ],
    outputs=[
        gr.Markdown(label="Verification Result"),
        gr.Markdown(label="Interpretation Guide")
    ],
    title="🧠 Signature Verifier (ORB + SSIM)",
    description="Compares two signatures using ORB keypoints and SSIM structure analysis. Tuned for real-world signature matching."
)

demo.launch(server_name="0.0.0.0", server_port=7860)

