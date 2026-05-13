from openai import OpenAI
from PIL import Image, ImageDraw
from pathlib import Path
import base64

API_BASE = "http://127.0.0.1:8007/v1"
API_KEY = "dms-qwen-secret"
MODEL = "qwen2.5-vl-7b"

img_path = Path("F:\\baoyantest\\dms\\test_vlm_image.png")

# 创建一张简单测试图，避免依赖外部图片
img = Image.new("RGB", (640, 360), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((40, 40, 600, 320), outline="black", width=4)
draw.text((80, 120), "DMS TEST 123", fill="black")
draw.text((80, 180), "Qwen2.5-VL image input check", fill="black")
img.save(img_path)

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
)

image_b64 = encode_image(img_path)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请读取图片中的主要文字，并说明你看到了什么。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ],
        }
    ],
    temperature=0,
    max_tokens=256,
)

print("image path:", img_path)
print("model output:")
print(resp.choices[0].message.content)