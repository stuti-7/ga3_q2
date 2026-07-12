import os
import json
import base64
from typing import Any
from fastapi import FastAPI, Request as FastAPIRequest
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

aipipe_token = os.getenv("AIPIPE_TOKEN", "")
client = OpenAI(api_key=aipipe_token, base_url="https://aipipe.org/openai/v1") if aipipe_token else None
DEFAULT_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4.1-mini")
FALLBACK_MODELS = list(dict.fromkeys([DEFAULT_MODEL, "gpt-4.1-mini", "gpt-4o-mini"]))


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
        payload: dict[str, Any] = {}

        if raw_body:
            payload = _parse_json_body(raw_body)
        else:
            try:
                form_data = await request.form()
                payload = {key: value for key, value in form_data.items()}
            except Exception:
                payload = {key: value for key, value in request.query_params.items()}

        image_base64 = str(payload.get("image_base64", ""))
        question = str(payload.get("question", ""))

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
            return {"error": "AIPIPE_TOKEN is not configured"}

        prompt = (
            f"{question}\n\n"
            "Answer with ONLY the direct answer, nothing else. "
            "No explanations, no full sentences, no markdown formatting. "
            "If the answer is a number, return just the number with no currency symbols, "
            "units, or commas (e.g. 4089.35)."
        )

        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

        last_error = None
        for model_name in FALLBACK_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                )
                answer = response.choices[0].message.content.strip()
                answer = answer.strip("*` \n")
                return {"answer": answer}
            except Exception as e:
                last_error = e
                if "not found" in str(e).lower() or "unsupported" in str(e).lower():
                    continue
                raise

        return {"error": f"AI Pipe request failed: {last_error}"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "ok"}
