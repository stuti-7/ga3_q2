import os
import json
import base64
from typing import Any
from fastapi import FastAPI, Request as FastAPIRequest
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

api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
FALLBACK_MODELS = [DEFAULT_MODEL, "gemini-2.0-flash", "gemini-2.0-flash-lite"]


class Request(BaseModel):
    image_base64: str
    question: str


def _sanitize_json_text(text: str) -> str:
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if ch == "\\" and not escaped:
            escaped = True
            out.append(ch)
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
            out.append(ch)
            escaped = False
            continue
        if in_string and ch in "\r\n\t":
            continue
        out.append(ch)
        escaped = False
    return "".join(out)


def _parse_json_body(raw_body: bytes) -> dict[str, Any]:
    text = raw_body.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Request body is empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        cleaned = _sanitize_json_text(text)
        parsed = json.loads(cleaned)

    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object")

    return parsed


def _normalize_base64_input(value: str) -> str:
    value = value.strip()
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return "".join(value.split())


@app.post("/answer-image")
async def answer(request: FastAPIRequest):
    try:
        raw_body = await request.body()
        payload = _parse_json_body(raw_body)
        image_base64 = payload.get("image_base64", "")
        question = payload.get("question", "")

        if not image_base64 or not question:
            return {"error": "Both image_base64 and question are required"}

        image_bytes = base64.b64decode(_normalize_base64_input(image_base64), validate=True)

        # Auto-detect mime type from magic bytes
        if image_bytes[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        elif image_bytes[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            mime_type = "image/webp"
        else:
            mime_type = "image/png"  # fallback

        if client is None:
            return {"error": "GEMINI_API_KEY is not configured"}

        last_error = None
        for model_name in FALLBACK_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        question,
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