"""OpenAI-compatible server for DiffusionVL-Qwen2.5VL-7B."""

from __future__ import annotations

import argparse
import base64
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_new_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str


class DiffusionVLRunner:
    def __init__(
        self,
        *,
        model_name: str,
        dtype: str,
        steps: int,
        gen_length: int,
        temperature: float,
    ) -> None:
        self.model_name = model_name
        self._dtype = dtype
        self._steps = steps
        self._gen_length = gen_length
        self._temperature = temperature
        self._model = None
        self._processor = None
        self._device = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "DiffusionVL server requires transformers + torch. "
                "Install with your preferred CUDA-enabled torch build."
            ) from exc

        dtype = None
        if self._dtype and self._dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if self._dtype not in dtype_map:
                raise ValueError(f"Unsupported dtype: {self._dtype}")
            dtype = dtype_map[self._dtype]
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = self._model.device

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("DiffusionVL model not loaded.")
        temp = self._temperature if temperature is None else temperature
        max_new = max_tokens or self._gen_length
        messages = _build_qwen_messages(prompt, images)
        try:
            chat = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            chat = prompt
        inputs = self._processor(text=chat, images=list(images), return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        do_sample = temp is not None and temp > 0
        generate_kwargs = {
            "max_new_tokens": max_new,
            "temperature": temp if do_sample else None,
            "do_sample": do_sample,
        }
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
        output_ids = self._model.generate(**inputs, **generate_kwargs)
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if chat and text.startswith(chat):
            text = text[len(chat) :].strip()
        return GenerationResult(text=text.strip(), model=self.model_name)


class DryRunRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        summary = f"dry-run ok: {len(prompt)} chars, {len(images)} images"
        return GenerationResult(text=summary, model=self.model_name)


def _decode_base64_image(value: str) -> Image.Image:
    if value.startswith("data:"):
        _, encoded = value.split(",", 1)
    else:
        encoded = value
    try:
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc


def _extract_messages(messages: Iterable[ChatMessage]) -> tuple[str, list[Image.Image]]:
    text_parts: list[str] = []
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text_parts.append(str(part.get("text") or ""))
                elif part_type == "image_url":
                    image_url = (part.get("image_url") or {}).get("url")
                    if not isinstance(image_url, str) or not image_url:
                        continue
                    if image_url.startswith("http://") or image_url.startswith("https://"):
                        raise HTTPException(
                            status_code=400,
                            detail="Only base64 image URLs are supported.",
                        )
                    images.append(_decode_base64_image(image_url))
        elif isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    prompt = "\n".join(part for part in text_parts if part.strip()).strip()
    return prompt, images


def _build_qwen_messages(prompt: str, images: Sequence[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for image in images:
        content.append({"type": "image", "image": image})
    return [{"role": "user", "content": content}]


def build_app(runner: Any) -> FastAPI:
    app = FastAPI(title="DiffusionVL-Qwen2.5VL-7B Server")

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        prompt, images = _extract_messages(request.messages)
        max_tokens = request.max_tokens or request.max_completion_tokens or request.max_new_tokens
        result = runner.generate(
            prompt,
            images,
            temperature=request.temperature,
            max_tokens=max_tokens,
        )
        now = int(time.time())
        return {
            "id": f"diffusionvl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
        }

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve DiffusionVL-Qwen2.5VL-7B locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model", default="hustvl/DiffusionVL-Qwen2.5VL-7B")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Skip model load for CI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = (
        DryRunRunner(args.model)
        if args.dry_run
        else DiffusionVLRunner(
            model_name=args.model,
            dtype=args.dtype,
            steps=args.steps,
            gen_length=args.gen_length,
            temperature=args.temperature,
        )
    )
    app = build_app(runner)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
