import json
import logging
import os
import pathlib

import httpx
import requests
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")

if not TOGETHER_API_KEY:
    raise RuntimeError("TOGETHER_API_KEY is not set in .env")

BASE_DIR = pathlib.Path(__file__).parent

app = FastAPI(title="Uncensored AI Backend")


def run_web_search(query: str, max_results: int = 4) -> list[str]:
    """Return a list of short text snippets from DuckDuckGo."""
    snippets: list[str] = []
    if not query.strip():
        return snippets

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                snippet = f"{title} — {body} ({href})"
                snippets.append(snippet)
    except Exception as exc:
        print(f"Search error: {exc!r}")
    return snippets
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = BASE_DIR / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Pydantic models ----------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    model: str | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    use_search: bool = False


class ChatResponse(BaseModel):
    reply: str
    history: list[ChatMessage]
    search_snippets: list[str] | None = None


class StreamChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    model: str | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    use_search: bool = False


class SearchRequest(BaseModel):
    query: str
    max_results: int = 3


class SearchResponse(BaseModel):
    results: list[str]


# ---------- Helper: call Together chat API ----------

def call_together_chat(
    message: str,
    history: list[ChatMessage],
    model: str | None,
    temperature: float | None,
    system_prompt: str | None,
    search_context: str | None = None,
) -> str:
    url = "https://api.together.xyz/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    chosen_model = model or "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temp = temperature if temperature is not None else 0.7

    base_prompt = (
        system_prompt
        or "You are an advanced AI assistant. Respond naturally and helpfully. Stay within legal and ethical boundaries."
    )

    if search_context:
        base_prompt += (
            "\n\nYou also have access to recent web search results for the user query.\n"
            "Use them as factual context when relevant, but still reason critically.\n"
            f"WEB_SEARCH_RESULTS:\n{search_context}"
        )

    base_system_msg = {
        "role": "system",
        "content": base_prompt,
    }

    msg_history = [{"role": m.role, "content": m.content} for m in history]
    msg_history = [base_system_msg] + msg_history + [
        {"role": "user", "content": message}
    ]

    data = {
        "model": chosen_model,
        "messages": msg_history,
        "max_tokens": 512,
        "temperature": temp,
    }

    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Together API error {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bad API response: {e}")


# ---------- FastAPI endpoints ----------


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(body: ChatRequest):
    search_snippets: list[str] | None = None

    if body.use_search:
        search_snippets = run_web_search(body.message, max_results=4)
        search_context = "\n\n".join(search_snippets)
    else:
        search_context = None

    reply = call_together_chat(
        body.message,
        body.history,
        body.model,
        body.temperature,
        body.system_prompt,
        search_context=search_context,
    )
    logging.info(f"/chat user message: {body.message[:100]!r}")
    logging.info(f"/chat reply: {reply[:100]!r}")

    new_history = body.history.copy()
    new_history.append(ChatMessage(role="user", content=body.message))
    new_history.append(ChatMessage(role="assistant", content=reply))

    return ChatResponse(reply=reply, history=new_history, search_snippets=search_snippets)


@app.post("/chat-stream")
async def chat_stream(body: StreamChatRequest):
    async def event_generator():
        async for chunk in together_stream_generator(body):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain")


async def together_stream_generator(
    body: StreamChatRequest,
) -> str:
    """Async generator that yields text chunks streamed from Together."""
    url = "https://api.together.xyz/v1/chat/completions"

    search_context = None
    if body.use_search:
        snippets = run_web_search(body.message, max_results=4)
        if snippets:
            search_context = "\n\n".join(snippets)

    chosen_model = body.model or "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    temp = body.temperature if body.temperature is not None else 0.7

    base_prompt = (
        body.system_prompt
        or "You are an advanced AI assistant. Respond naturally and helpfully. Avoid anything illegal or genuinely harmful."
    )

    if search_context:
        base_prompt += (
            "\n\nYou also have access to recent web search results for the user query.\n"
            "Use them as factual context when relevant, but still reason critically.\n"
            f"WEB_SEARCH_RESULTS:\n{search_context}"
        )

    system_msg = {"role": "system", "content": base_prompt}
    msg_history = [{"role": m.role, "content": m.content} for m in body.history]
    msg_history = [system_msg] + msg_history + [
        {"role": "user", "content": body.message}
    ]

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": chosen_model,
        "messages": msg_history,
        "max_tokens": 512,
        "temperature": temp,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue

                if line.startswith("data:"):
                    data_part = line[len("data:"):].strip()
                else:
                    data_part = line.strip()

                if not data_part:
                    continue
                if data_part == "[DONE]":
                    break

                try:
                    obj = json.loads(data_part)
                    delta = obj["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue


@app.post("/search", response_model=SearchResponse)
def search_endpoint(body: SearchRequest):
    results: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(body.query, max_results=body.max_results):
            snippet = f"{r.get('title', '')} - {r.get('body', '')} ({r.get('href', '')})"
            results.append(snippet)
    return SearchResponse(results=results)


# Placeholder for /generate-image (we'll wire this later)
class ImageRequest(BaseModel):
    prompt: str


class ImageResponse(BaseModel):
    url: str


@app.post("/generate-image", response_model=ImageResponse)
def generate_image(body: ImageRequest):
    if not IMAGE_API_KEY:
        raise HTTPException(status_code=500, detail="IMAGE_API_KEY is not set")

    url = "https://stablediffusionapi.com/api/v4/dreambooth"
    payload = {
        "key": IMAGE_API_KEY,
        "prompt": body.prompt,
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "samples": 1,
        "num_inference_steps": "30",
        "enhance_prompt": "yes",
        "scheduler": "UniPCMultistepScheduler",
    }

    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Image API error {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    try:
        image_url = data["output"][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bad image API response: {e}")

    return ImageResponse(url=image_url)
