"""
infer.py
--------
Loads trained U-Net weights and runs inference on new images.
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from demo import make_synthetic_scan
from medical_coseg.unet import UNet
from medical_coseg.postprocessing import DEFAULT_PALETTE

def infer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and load weights
    model = UNet(n_channels=3, n_classes=5).to(device)
    weights_path = "outputs/unet_weights.pth"
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Loaded trained U-Net weights.")
    
    # Create test images
    print("Generating test images...")
    test_images = [make_synthetic_scan(256, 256, seed=i) for i in range(100, 103)]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("U-Net Prediction")
    
    for i, bgr_img in enumerate(test_images):
        # Preprocess
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        input_tensor = transform(img_resized).unsqueeze(0).to(device)
        
        # Infer
        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        # Colorize prediction
        color_map = np.zeros((224, 224, 3), dtype=np.uint8)
        for k, color in enumerate(DEFAULT_PALETTE[:5]):
            # matplotlib expects RGB
            color_rgb = (color[2], color[1], color[0]) 
            color_map[prediction == k] = color_rgb
            
        # Plot
        axes[i, 0].imshow(img_resized)
        axes[i, 0].axis('off')
        
        # Blend overlay
        overlay = cv2.addWeighted(img_resized, 0.5, color_map, 0.5, 0)
        axes[i, 1].imshow(overlay)
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    out_path = "outputs/unet_predictions.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"\nSaved inference visualization to: {out_path}")

if __name__ == "__main__":
    infer()
