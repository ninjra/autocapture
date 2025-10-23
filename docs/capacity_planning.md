# Capacity Planning

## Daily and Annual Storage

Assumptions:
- 8 hours of active HID input per day within a 14-hour work window.
- Continuous capture at 3 fps while input is active.
- 50% duplicate suppression through perceptual hashing and window metadata.
- WebP encoding at visually lossless settings (≈1.8 MB per 8K frame).
- OCR metadata and embeddings per capture ≈ 18 KB combined.

| Metric | Value |
| --- | --- |
| Frames captured per active day | ~43,000 |
| Daily image storage | ~77 GB |
| Daily metadata storage | ~0.8 GB |
| Annual image storage (365 days) | ~28 TB |
| Annual storage with 50% dedupe | ~14 TB |
| Storage at 2 fps soft cap (50% dedupe) | ~9 TB/year |

### Recommendations
- Enforce a 2 fps soft cap for extended sessions with minimal UI change.
- Retain full-resolution images for 90 days; archive OCR/metadata indefinitely.
- Monitor NAS usage via Prometheus alerting at 2.6 TB to trigger pruning or capacity upgrades.

## GPU Utilization Targets
- **Capture encoding:** NVENC keeps per-frame encode time under 20 ms.
- **OCR batches:** 32-image batches processed within ~2.5 s on RTX 4090 (PaddleOCR CUDA).
- **Embedding batches:** 256-span nightly jobs complete in under 5 minutes using CUDA-accelerated SentenceTransformer.

Keep GPU scheduling windows outside peak CPU-bound tasks to avoid contention.
