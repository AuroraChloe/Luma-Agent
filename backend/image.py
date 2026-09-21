import os
from image_provider import edit_image as provider_edit_image, generate_image as provider_generate_image

OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "120"))
IMAGE_GENERATION_TIMEOUT_SECONDS = int(os.getenv("IMAGE_GENERATION_TIMEOUT_SECONDS", "120"))
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://127.0.0.1:3000/v1")


def image_generate(prompt, model='gpt-image-2', n=1):
    return provider_generate_image(prompt, model=model, n=n)

def image_edit(image_path, prompt, model='gpt-image-2', n=1):
    return provider_edit_image([image_path], prompt, model=model, n=n)

# image_path = "/opt/key_college/temp_images/1f0359c0985d4baa951fbec8b1cbd5d8.png"
# prompt = "把图片整体改成绿色风格"

# result = image_edit(image_path, prompt)

# print(result["data"][0]["url"])
