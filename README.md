# 🧠 Signature Verifier (ORB + SSIM)

## Setup

### 1. Create and activate virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


2. Install dependencies

pip install -r requirements.txt

3. Run the App

python app.py

4. Open in browser

Navigate to: http://localhost:7860
Description

This tool compares two signature images using:

    ORB keypoint matching

    SSIM (Structural Similarity Index)

Thresholds (Tuned for Real-world Data)

    ORB Match Ratio: > 0.04

    SSIM Score: > 0.74


---

### ✅ 4. **Create a `Dockerfile` (Optional for Deployment)**

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app



RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]

    ✅ You can now build and run:

docker build -t signature-verifier .
docker run -p 7860:7860 signature-verifier

✅ 5. Package into a Zip File

To distribute:
zip -r signature-verifier.zip app.py requirements.txt README.md assets/


