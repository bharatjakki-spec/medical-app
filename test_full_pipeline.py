"""
test_full_pipeline.py
---------------------
Runs the complete end-to-end ML & Gemini diagnosis pipeline on a synthetic scan.
"""

import cv2
import torch
import sys
import os
from PIL import Image
import io
from google import genai

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from demo import make_synthetic_scan
from app import load_model, run_segmentation, GEMINI_API_KEY

def main():
    print("=" * 60)
    print("  Running Full Medical AI Pipeline Test")
    print("=" * 60)

    # 1. Generate synthetic medical image
    print("\n[1/3] Generating test synthetic scan...")
    bgr_img = make_synthetic_scan(256, 256, seed=42)
    cv2.imwrite("outputs/test_input.png", bgr_img)
    print("      Saved test input to outputs/test_input.png")

    # 2. Run U-Net Segmentation
    print("\n[2/3] Running U-Net Structural Segmentation...")
    model, device = load_model()
    if model is None:
        print("      U-Net model weights not found! Running train.py first...")
        from train import generate_pseudo_labels, train_unet
        origs, masks = generate_pseudo_labels(n_scans=10, n_clusters=5)
        train_unet(origs, masks, n_classes=5, epochs=5)
        model, device = load_model()

    img_resized, overlay, prediction = run_segmentation(bgr_img, model, device)
    cv2.imwrite("outputs/test_segmentation_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("      Saved segmentation overlay to outputs/test_segmentation_overlay.png")

    # 3. Call AI Diagnosis & Treatment Plan
    print("\n[3/3] Generating AI Diagnosis & Treatment Plan...")
    import os
    gemini_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    
    prompt = (
        "You are an expert AI medical doctor. Please analyze this medical scan (X-ray/MRI). "
        "1. Describe any potential anomalies or structures you observe.\n"
        "2. Suggest a highly plausible diagnosis based on standard medical knowledge.\n"
        "3. Provide a detailed, standard treatment plan and next steps for this diagnosis."
    )

    report_text = None
    if gemini_key:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[pil_img, prompt]
            )
            report_text = response.text
        except Exception as e:
            print(f"      Gemini API notice: {e}")

    if report_text is None:
        try:
            from app import get_diagnosis_openai, DEFAULT_OPENAI_KEY
            _, raw_bytes = cv2.imencode('.png', bgr_img)
            report_text = get_diagnosis_openai(DEFAULT_OPENAI_KEY, raw_bytes.tobytes(), prompt)
        except Exception as e:
            print(f"      OpenAI API notice: {e}")
            from app import generate_medical_report
            report_text = generate_medical_report(prediction)

    print("\n" + "=" * 60)
    print("  AI DIAGNOSIS & TREATMENT REPORT")
    print("=" * 60 + "\n")
    print(report_text)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
