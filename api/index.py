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

        response = client.models.generate_content(
            model="gemini-2.0-flash",
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
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "ok"}