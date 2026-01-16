# TRON Format Support

Autocapture supports TRON (Token Reduced Object Notation) as a compact JSON superset
for context packs and structured answers. JSON remains valid TRON; TRON is an
optional, more token-efficient format for large structured payloads.

## Usage

- `output.format`: `text`, `json`, or `tron`.
- `output.context_pack_format`: `json` or `tron`.
- `autocapture/format/tron.py` provides deterministic encode/decode utilities for
  the subset we emit (uniform arrays and simple objects).

## References

- TRON format overview: https://tron-format.github.io/
- Format comparison explainer: https://www.piotr-sikora.com/blog/2025-12-05-toon-tron-csv-yaml-json-format-comparison
