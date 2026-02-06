import requests
import base64
from requests.adapters import HTTPAdapter

API_URL = "http://localhost:8000/api/v2/check_quality"
DESKTOP_PATH = "/home/chetan/Desktop"
# Using same images since logic.py no longer blocks them, 
# but check_quality should block them.
IMAGES = [
    "distorted1.jpeg", # Valid (should succeed)
    "distorted2.webp", # Bad (Fail anatomy)
    "distorted3.jpg",  # Bad (Fail conf)
    "dist4.jpeg"       # Bad (Fail anatomy)
]

def get_base64_data_uri(file_path):
    with open(f"{DESKTOP_PATH}/{file_path}", "rb") as f:
        data = base64.b64encode(f.read()).decode('utf-8')
        ext = file_path.split('.')[-1]
        media_type = f"image/{ext}" if ext != 'jpg' else 'image/jpeg'
        return f"data:{media_type};base64,{data}"

print("\n🧪 Testing Dedicated Image Quality API\n")

for img_name in IMAGES:
    print(f"📸 Checking {img_name}...")
    try:
        # Simulate S3 URL by passing Base64 Data URI (server supports it)
        url = get_base64_data_uri(img_name)
        
        response = requests.post(API_URL, json={"imageUrl": url}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            status = "✅ PASS" if data['success'] else "❌ REJECT"
            print(f"   => Status: {status}")
            print(f"   => Message: {data['message']}")
            print(f"   => Score: {data['quality_score']}")
            if data['error']:
                print(f"   => Error: {data['error']}")
        else:
            print(f"   ⚠️ API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   ⚠️ Exception: {e}")
    print("-" * 50)
