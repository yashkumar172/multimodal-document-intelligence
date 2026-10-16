import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("FAILED - GEMINI_API_KEY not set")
    sys.exit(1)

try:
    client = genai.Client(api_key=api_key)
    print("Key accepted. Models supporting generateContent:")
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(" -", m.name)
except Exception as e:
    print("FAILED -", type(e).__name__, str(e))
    sys.exit(1)
