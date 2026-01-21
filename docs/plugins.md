# Plugins

Autocapture uses a plugin-first system for providers and pipeline stages. Plugins are discovered
without importing code and must be explicitly enabled before their extensions are usable.

## Terminology
- **Plugin**: a package or folder with `plugin.yaml`.
- **Extension**: a specific capability (e.g., `llm.provider:openai`).

## Discovery
Plugins are discovered from:
1. Built-ins bundled with Autocapture.
2. Folder plugins at `${capture.data_dir}/plugins`.
3. Entry-point plugins under the `autocapture.plugins` group.

Discovery does **not** import plugin code.

## Enable / Disable
External plugins require explicit enablement and hash acceptance.

CLI:
```bash
poetry run autocapture plugins list
poetry run autocapture plugins enable <plugin_id> --accept-hashes
poetry run autocapture plugins disable <plugin_id>
poetry run autocapture plugins lock <plugin_id>   # re-approve hashes
poetry run autocapture plugins doctor
```

API:
- `GET /api/plugins/catalog`
- `GET /api/plugins/extensions?kind=llm.provider`
- `POST /api/plugins/enable`
- `POST /api/plugins/disable`
- `POST /api/plugins/lock`
- `GET /api/plugins/health`

UI:
Open the **Plugins** tab to enable/disable plugins and review hashes.

## Settings Structure
Plugin state is stored in `settings.json`:
```json
{
  "plugins": {
    "enabled": ["vendor.plugin_id"],
    "disabled": [],
    "extension_overrides": {
      "llm.provider:openai": "vendor.plugin_id"
    },
    "locks": {
      "vendor.plugin_id": {
        "manifest_sha256": "...",
        "code_sha256": "...",
        "accepted_at_utc": "2026-01-19T00:00:00Z"
      }
    },
    "configs": {
      "vendor.plugin_id": {
        "custom_key": "value"
      }
    }
  }
}
```

## Safety Rules
Core-owned policy gates enforce:
- `privacy.cloud_enabled=true` and `offline=false` for cloud text providers.
- `privacy.allow_cloud_images=true` for cloud image providers.
- Stage-level `allow_cloud` flags when a stage context is present.

Plugins cannot bypass these rules.

## Deterministic Resolution
If two enabled plugins provide the same extension ID, resolution fails unless
`plugins.extension_overrides` specifies the winning plugin ID.

## Graph adapters
Graph retrieval adapters are exposed via `graph.adapter` extensions. Built-in adapters
wrap the configured HTTP clients for `graphrag`, `hypergraphrag`, and `hyperrag`.
These adapters still use `retrieval.graph_adapters.*` config for endpoints and timeouts.

## Table extractors
Table extraction providers are exposed via `table.extractor` extensions. Enable the
pipeline with `table_extractor.enabled=true` and select a provider with
`routing.table_extractor`. Cloud-backed extractors are additionally gated by
`table_extractor.allow_cloud` and privacy/offline policy gates.

## Decode backends
Decode backends are exposed via `decode.backend` extensions (swift/lookahead/medusa).
They are proxy adapters that forward requests to OpenAI-compatible servers. Configure
per-backend endpoints in `settings.json` under `plugins.configs`:
```json
{
  "plugins": {
    "configs": {
      "autocapture.builtin.decode": {
        "backends": {
          "medusa": {
            "base_url": "http://127.0.0.1:8012",
            "allow_cloud": false,
            "max_concurrency": 1
          }
        }
      }
    }
  }
}
```
Use the backend id in `model_registry.stages[*].decode.backend_provider_id`.
Example (vLLM, local):
```bash
scripts/run_vllm_gpu_a.sh <model-name>
scripts/run_vllm_gpu_b.sh <model-name>
scripts/run_vllm_cpu.sh <model-name>
```

## Training pipelines
Training pipelines are exposed via `training.pipeline` extensions (lora/qlora/dpo).
By default they return a structured "unavailable" response. You can opt in to
command-based execution via `settings.json`:
```json
{
  "plugins": {
    "configs": {
      "autocapture.builtin.training": {
        "pipelines": {
          "lora": {
            "command": ["python", "scripts/run_lora.py", "--dataset", "{dataset_path}"],
            "args": ["--out", "{output_dir}", "--params", "{params_json}"],
            "working_dir": ".",
            "timeout_s": 3600
          }
        }
      }
    }
  }
}
```
Placeholders: `{dataset_path}`, `{output_dir}`, `{run_id}`, `{params_json}`.
CLI:
```bash
poetry run autocapture training list
poetry run autocapture training run lora --config path/to/run.yml
```
To execute a training step, pass `--train` and provide a local model path plus optional
dependencies (`transformers`, `peft`, `torch`, `trl` for DPO, `bitsandbytes` for QLoRA).
Example configs (settings.json):
```json
{
  "plugins": {
    "configs": {
      "autocapture.builtin.training": {
        "pipelines": {
          "qlora": {
            "command": ["python", "scripts/run_qlora.py", "--dataset", "{dataset_path}"],
            "args": ["--out", "{output_dir}", "--params", "{params_json}"],
            "timeout_s": 3600
          },
          "dpo": {
            "command": ["python", "scripts/run_dpo.py", "--dataset", "{dataset_path}"],
            "args": ["--out", "{output_dir}", "--params", "{params_json}"],
            "timeout_s": 3600
          }
        }
      }
    }
  }
}
```

## Safe Mode
Set `AUTOCAPTURE_SAFE_MODE=1` or `plugins.safe_mode=true` to load built-ins only.
