# Architecture Notes

## Target Environment
- Windows 11 host capture + local UI.
- WSL/Linux backend for DB/indexing/worker loops.
- RTX 4090 (CUDA) with local-first services (Ollama, RapidOCR + ONNX CUDA, local embeddings, local Qdrant).

## Phase 2 Focus
- Retrieval reranking + query refinement prompts.
- Key portability via encrypted export/import.
- PromptOps sandboxing.
- API paging.
- RapidOCR provider hardening.
- Default retention 60 days.
- WSL bridge endpoints + dashboard storage widget (if missing).
