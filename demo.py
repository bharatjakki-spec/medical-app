"""
demo.py
-------
Self-contained demonstration of Medical Image Co-Segmentation.
Generates synthetic medical-style images (no real scans required).

Run:
    python demo.py

Output (saved to outputs/):
    - scan_001_segmentation.png ... scan_006_segmentation.png
    - cosegmentation_grid.png
    - cluster_stats.png
"""

import numpy as np
import cv2
from pipeline import MedicalCoSegPipeline


# -----------------------------------------------------------------------
# Synthetic medical image generator
# -----------------------------------------------------------------------

def make_synthetic_scan(
    height: int = 256,
    width: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a realistic-looking grayscale synthetic medical scan
    with multiple circular/elliptical "structures" (like tissue regions
    or anomalies) embedded in a noisy background.

    Returns a BGR uint8 image.
    """
    rng = np.random.RandomState(seed)

    # Background: spatially-varying Gaussian noise (simulate tissue texture)
    bg = rng.randint(30, 80, (height, width), dtype=np.uint8)

    # Blur for smooth base
    bg = cv2.GaussianBlur(bg, (15, 15), 5)

    canvas = bg.copy().astype(np.float32)

    # Draw 3-6 organic blob structures (simulate organs / lesions)
    n_blobs = rng.randint(3, 7)
    for _ in range(n_blobs):
        cx = rng.randint(30, width - 30)
        cy = rng.randint(30, height - 30)
        rx = rng.randint(15, 60)
        ry = rng.randint(15, 60)
        angle = rng.randint(0, 180)
        intensity = rng.randint(100, 220)
        noise_strength = rng.randint(10, 40)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
        mask_f = mask.astype(np.float32) / 255.0

        blob_texture = (
            np.ones((height, width)) * intensity
            + rng.randn(height, width) * noise_strength
        ).clip(0, 255)

        canvas = canvas * (1 - mask_f) + blob_texture * mask_f

    # Add global Gaussian noise
    noise = rng.randn(height, width) * 8
    canvas = (canvas + noise).clip(0, 255).astype(np.uint8)

    # Smooth final image
    canvas = cv2.GaussianBlur(canvas, (3, 3), 1)

    # Convert grayscale -> BGR (required by our preprocessor)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return bgr


# -----------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------

def main():
    print("\n" + "#"*60)
    print("#  Medical Image Co-Segmentation  -  DEMO")
    print("#  Generating 6 synthetic scans and running pipeline ...")
    print("#"*60 + "\n")

    # Generate 6 synthetic "patient scans" with different random seeds
    N_SCANS = 6
    print(f"Generating {N_SCANS} synthetic medical scans ...")
    bgr_arrays = [make_synthetic_scan(256, 256, seed=i) for i in range(N_SCANS)]

    # Save raw synthetic images for reference
    import os
    os.makedirs("outputs/raw", exist_ok=True)
    for i, img in enumerate(bgr_arrays):
        cv2.imwrite(f"outputs/raw/synthetic_scan_{i+1:03d}.png", img)
    print(f"Saved raw synthetic scans to outputs/raw/\n")

    # Initialise and run the co-segmentation pipeline
    pipeline = MedicalCoSegPipeline(
        n_clusters=5,          # discover 5 semantic tissue regions
        target_size=(224, 224),
        feature_size=(56, 56),
        use_pca=True,
        pca_components=64,
        use_minibatch=True,
        overlay_alpha=0.55,
        output_dir="outputs",
        use_gpu=True,          # falls back to CPU if CUDA unavailable
    )

    results = pipeline.run_on_arrays(bgr_arrays)

    print("\nOutput files:")
    for p in results["output_paths"]:
        print(f"  -> {p}")

    print("\n" + "#"*60)
    print("#  Demo complete! Open outputs/ to see the results.")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
