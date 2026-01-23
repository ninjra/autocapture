"""Warm local models to avoid first-run stalls."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

from autocapture.config import load_config
from autocapture.paths import default_config_path
from PIL import Image


def warm_embeddings(config) -> bool:
    if config.routing.embedding.strip().lower() == "disabled":
        print("embeddings: skip (routing disabled)")
        return False
    if config.embed.text_model == "disabled":
        print("embeddings: skip (model disabled)")
        return False
    if config.offline:
        print("embeddings: skip (offline mode)")
        return False
    from autocapture.embeddings.service import EmbeddingService

    service = EmbeddingService(config.embed)
    _ = service.embed_texts(["warmup"])
    service.close()
    print(f"embeddings: ok ({config.embed.text_model})")
    return True


def warm_sparse(config) -> bool:
    if not config.retrieval.sparse_enabled:
        print("sparse: skip (disabled)")
        return False
    model = config.retrieval.sparse_model
    if model in {"hash", "hash-splade", "local-test"}:
        print(f"sparse: skip ({model})")
        return False
    from autocapture.embeddings.sparse import SparseEncoder

    encoder = SparseEncoder(model)
    _ = encoder.encode(["warmup"])
    print(f"sparse: ok ({model})")
    return True


def warm_reranker(config) -> bool:
    if config.routing.reranker.strip().lower() != "enabled" or not config.reranker.enabled:
        print("reranker: skip (disabled)")
        return False
    from autocapture.memory.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(config.reranker)
    _ = reranker.rank("warmup", ["document"])
    reranker.close()
    print(f"reranker: ok ({config.reranker.model})")
    return True


def warm_ocr(config) -> bool:
    if config.ocr.engine.strip().lower() == "disabled":
        print("ocr: skip (disabled)")
        return False
    from autocapture.vision.rapidocr import RapidOCRExtractor

    extractor = RapidOCRExtractor(config.ocr)
    extractor.close()
    print("ocr: ok (rapidocr)")
    return True


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _warm_vision_backend(
    *,
    name: str,
    config,
    backend,
    allow_cloud: bool,
) -> None:
    from autocapture.vision.clients import VisionClient
    from autocapture.vision.extractors import _cloud_images_allowed, _resolve_backend
    from autocapture.llm.prompt_strategy import PromptStrategySettings

    provider, base_url, api_key, model = _resolve_backend(config, backend)
    if not _cloud_images_allowed(config, base_url, provider, allow_cloud):
        raise RuntimeError("cloud vision calls not permitted by privacy settings")

    image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    client = VisionClient(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_s=config.llm.timeout_s,
        retries=config.llm.retries,
        prompt_strategy=PromptStrategySettings.from_llm_config(
            config.llm, data_dir=config.capture.data_dir
        ),
        governor=None,
        priority="background",
    )
    _ = client.generate("You are a test system.", "Reply with OK.", [image_bytes])
    print(f"vlm:{name}: ok ({provider} {model})")


def _normalize_engine(value: str | None) -> str:
    return (value or "").strip().lower()


def _is_vlm_engine(engine: str) -> bool:
    return engine in {"vlm", "qwen-vl"}


def _is_deepseek_engine(engine: str) -> bool:
    return engine in {"deepseek-ocr", "deepseek"}


def warm_vlms(config) -> bool:
    if not _env_truthy("AUTOCAPTURE_WARM_VLM", True):
        print("vlm: skip (AUTOCAPTURE_WARM_VLM=0)")
        return False
    warmed = False
    allow_cloud = True
    errors: list[str] = []

    engine = _normalize_engine(config.vision_extract.engine)
    fallback = _normalize_engine(config.vision_extract.fallback_engine)
    if _is_vlm_engine(engine) or _is_vlm_engine(fallback):
        try:
            _warm_vision_backend(
                name="vision_extract.vlm",
                config=config,
                backend=config.vision_extract.vlm,
                allow_cloud=allow_cloud,
            )
            warmed = True
        except Exception as exc:
            errors.append(f"vision_extract.vlm: {exc}")

    if _is_deepseek_engine(engine) or _is_deepseek_engine(fallback):
        try:
            _warm_vision_backend(
                name="vision_extract.deepseek_ocr",
                config=config,
                backend=config.vision_extract.deepseek_ocr,
                allow_cloud=allow_cloud,
            )
            warmed = True
        except Exception as exc:
            errors.append(f"vision_extract.deepseek_ocr: {exc}")

    if config.vision_extract.ui_grounding.enabled:
        try:
            _warm_vision_backend(
                name="vision_extract.ui_grounding",
                config=config,
                backend=config.vision_extract.ui_grounding.vlm,
                allow_cloud=allow_cloud,
            )
            warmed = True
        except Exception as exc:
            errors.append(f"vision_extract.ui_grounding: {exc}")

    if config.agents.enabled:
        try:
            backend = config.agents.vision
            # Treat agent vision as a VLM backend.
            provider = backend.provider
            base_url = backend.base_url or config.llm.ollama_url
            api_key = backend.api_key
            model = backend.model
            from autocapture.vision.clients import VisionClient
            from autocapture.vision.extractors import _cloud_images_allowed
            from autocapture.llm.prompt_strategy import PromptStrategySettings

            if not _cloud_images_allowed(config, base_url, provider, allow_cloud):
                raise RuntimeError("cloud vision calls not permitted by privacy settings")
            image = Image.new("RGB", (16, 16), color=(128, 128, 128))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            client = VisionClient(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout_s=config.llm.timeout_s,
                retries=config.llm.retries,
                prompt_strategy=PromptStrategySettings.from_llm_config(
                    config.llm, data_dir=config.capture.data_dir
                ),
                governor=None,
                priority="background",
            )
            _ = client.generate("You are a test system.", "Reply with OK.", [image_bytes])
            print(f"vlm:agents.vision: ok ({provider} {model})")
            warmed = True
        except Exception as exc:
            errors.append(f"agents.vision: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))
    if not warmed:
        print("vlm: skip (no VLM backends enabled)")
        return False
    return True


def _collect_ollama_model_map(config) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    def _add(path: str, model: str | None, kind: str) -> None:
        if not model:
            return
        entries.append({"path": path, "model": model, "kind": kind})

    if config.llm.provider == "ollama":
        _add("llm.ollama_model", config.llm.ollama_model, "text")

    if config.agents.enabled and config.agents.vision.provider == "ollama":
        _add("agents.vision.model", config.agents.vision.model, "vision")

    engine = _normalize_engine(config.vision_extract.engine)
    fallback = _normalize_engine(config.vision_extract.fallback_engine)
    use_vlm = _is_vlm_engine(engine) or _is_vlm_engine(fallback)
    use_deepseek = _is_deepseek_engine(engine) or _is_deepseek_engine(fallback)

    if use_vlm and config.vision_extract.vlm.provider == "ollama":
        _add("vision_extract.vlm.model", config.vision_extract.vlm.model, "vision")
    if use_deepseek and config.vision_extract.deepseek_ocr.provider == "ollama":
        _add("vision_extract.deepseek_ocr.model", config.vision_extract.deepseek_ocr.model, "vision")
    if config.vision_extract.ui_grounding.enabled:
        if config.vision_extract.ui_grounding.vlm.provider == "ollama":
            _add("vision_extract.ui_grounding.vlm.model", config.vision_extract.ui_grounding.vlm.model, "vision")

    return entries


def _collect_ollama_models(config) -> list[str]:
    models: list[str] = []
    for entry in _collect_ollama_model_map(config):
        name = entry.get("model")
        if not name:
            continue
        if name not in models:
            models.append(name)
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ollama-models",
        action="store_true",
        help="Print comma-separated Ollama models required by config.",
    )
    parser.add_argument(
        "--ollama-models-json",
        action="store_true",
        help="Print JSON list of Ollama model entries with path and kind.",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip warming VLM backends (default is warm).",
    )
    args = parser.parse_args()

    config_path = os.environ.get("AUTOCAPTURE_CONFIG", str(default_config_path()))
    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"warm_models: failed to load config: {exc}")
        return 1

    if args.ollama_models_json:
        print(json.dumps(_collect_ollama_model_map(config)))
        return 0
    if args.ollama_models:
        print(",".join(_collect_ollama_models(config)))
        return 0

    print(f"warm_models: config={config_path}")
    ok = True
    for fn in (warm_embeddings, warm_sparse, warm_reranker, warm_ocr):
        try:
            fn(config)
        except Exception as exc:
            ok = False
            print(f"{fn.__name__}: failed: {exc}")
    if args.skip_vlm:
        os.environ["AUTOCAPTURE_WARM_VLM"] = "0"
    try:
        warm_vlms(config)
    except Exception as exc:
        ok = False
        print(f"warm_vlms: failed: {exc}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
