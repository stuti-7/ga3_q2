import os
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
FALLBACK_MODELS = [DEFAULT_MODEL, "gemini-2.0-flash", "gemini-2.0-flash-lite"]


class Request(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
def answer(req: Request):
    try:
        image_bytes = base64.b64decode(req.image_base64)

        # Auto-detect mime type from magic bytes
        if image_bytes[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        elif image_bytes[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            mime_type = "image/webp"
        else:
            mime_type = "image/png"  # fallback

        last_error = None
        for model_name in FALLBACK_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        req.question,
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                    ],
                )
                return {"answer": response.text.strip()}
            except Exception as e:
                last_error = e
                if "not found" in str(e).lower() or "unsupported" in str(e).lower():
                    continue
                raise

        return {"error": f"Gemini request failed: {last_error}"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "ok"}