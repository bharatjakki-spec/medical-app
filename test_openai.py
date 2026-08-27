"""
test_openai.py
--------------
Tests the new OpenAI API key provided by the user.
"""

import base64
import cv2
from openai import OpenAI
from demo import make_synthetic_scan

OPENAI_API_KEY = "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"

def main():
    print("Testing OpenAI API key...")
    bgr_img = make_synthetic_scan(256, 256, seed=42)
    _, buffer = cv2.imencode('.png', bgr_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a medical AI assistant. Briefly analyze this scan and provide a 2-sentence diagnosis sample."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=200,
    )
    print("\n" + "=" * 60)
    print("  OPENAI DIAGNOSIS SAMPLE OUTPUT")
    print("=" * 60 + "\n")
    print(response.choices[0].message.content)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
