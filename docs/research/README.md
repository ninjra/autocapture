# Research Scout

The research scout surfaces recent models and papers related to screen understanding,
OCR, reranking, and embeddings. It fetches Hugging Face model metadata and arXiv
papers, then writes a ranked report plus an append-only log.

## Watchlist

- Hugging Face tags: vision-language, ocr, reranker, embeddings.
- arXiv keywords: "vision-language", "document understanding", "reranking",
  "screen understanding", "diffusion transformer", "prompt repetition",
  "prompt duplication". Multi-word phrases are treated as exact phrases and
  expanded into required-token matches after stopword filtering (e.g., "prompt
  repetition" matches the phrase or papers that mention both prompt and
  repetition anywhere in the text).

## Interesting Now

- https://arxiv.org/abs/2509.26507
- https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B
- https://github.com/nathan-barry/tiny-diffusion
- https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/

## Usage

```powershell
poetry run autocapture research scout --out "docs/research/scout_report.json"
```

The command appends a summary to `docs/research/scout_log.md` by default.

## Automation

The scheduled workflow `.github/workflows/research-scout.yml` runs the scout on a
cron schedule and opens a PR only if the ranked list changes beyond the configured
threshold.
