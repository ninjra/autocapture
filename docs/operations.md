# Operations Guide

## Local Model Serving (Vision + LLM)

Autocapture defaults to local-only model endpoints. Recommended local options:

- **Ollama (local)** for Qwen2.5-VL VLM extraction and fast drafts. Configure
  `vision_extract.vlm.provider: ollama` and `model: qwen2.5-vl:7b-instruct`.
- **OpenAI-compatible local servers** (vLLM, llama.cpp, Open WebUI) for larger
  Qwen2.5-VL or DiffusionVL variants. Configure `vision_extract.vlm.provider:
  openai_compatible`, `base_url: http://127.0.0.1:PORT`, and the model name.
- **DiffusionVL local server** (scripted OpenAI-compatible endpoint):
  ```powershell
  poetry run python tools/diffusionvl_server.py --host 127.0.0.1 --port 8010
  ```
  Install `transformers` plus a CUDA-enabled PyTorch build before running real mode.
  Then set:
  ```yaml
  vision_extract:
    vlm:
      provider: "openai_compatible"
      base_url: "http://127.0.0.1:8010"
      model: "hustvl/DiffusionVL-Qwen2.5VL-7B"
  ```
  Use `--dry-run` to validate routes in CI without loading the model.
- **DiffusionVL-Qwen2.5VL-7B** is supported as a selectable model (use an
  OpenAI-compatible server if available): https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B
- **DeepSeek-OCR** is an optional backend (`vision_extract.engine: deepseek-ocr`):
  https://huggingface.co/deepseek-ai/DeepSeek-OCR and
  https://github.com/deepseek-ai/DeepSeek-OCR

Cloud vision is blocked by default. To enable (explicit opt-in):

1. `privacy.cloud_enabled: true`
2. `privacy.allow_cloud_images: true`
3. `vision_extract.vlm.allow_cloud: true` (or `vision_extract.deepseek_ocr.allow_cloud: true`)

Acceleration references:

- NVIDIA RTX local acceleration: https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/
- Tiny diffusion experiments: https://github.com/nathan-barry/tiny-diffusion

## Doctor + Windows paths

If `autocapture doctor` reports a writable-path failure on Windows, confirm
`LOCALAPPDATA` is set (Git Bash/MSYS sometimes omits it) or explicitly set
`capture.data_dir`/`capture.staging_dir` in your config.

## Model Stages (Routing)

Use `model_stages` to route query refinement, draft generation, final answer, and tool
transform stages. Each stage can override provider/model/base_url and requires
`allow_cloud: true` for any non-local endpoint.

## Output Formats (JSON/TRON)

`output.format` controls answer serialization (`text`, `json`, or `tron`). The context
pack payload sent to LLMs is controlled by `output.context_pack_format` (`json` or
`tron`). For cloud stages, TRON context packs are used only when
`output.allow_tron_compression=true` (default `false`).

## Decode Backends (Swift/Lookahead/Medusa)

Decode backends are proxy adapters that point at OpenAI-compatible servers. Launch vLLM:
```bash
scripts/run_vllm_gpu_a.sh <model-name>
scripts/run_vllm_gpu_b.sh <model-name>
scripts/run_vllm_cpu.sh <model-name>
```
Then configure the decode backend in `settings.json`:
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
Reference the backend in `model_registry.stages[*].decode.backend_provider_id`.

## Research Scout

Generate a cached model/paper report (local-first, offline-aware):

```powershell
poetry run autocapture research scout --out "docs/research/scout_report.json"
```

The scout appends a short summary to `docs/research/scout_log.md` by default.
Scheduled runs are available via `.github/workflows/research-scout.yml`, which
opens a PR only when the ranked list changes beyond the configured threshold.

## Training Pipelines (Scaffold)

Training pipelines are scaffolded as plugins only; they do not run real training
jobs until you supply an implementation.

```bash
poetry run autocapture training list
poetry run autocapture training run lora --config path/to/run.yml
```
Use `--dry-run` to validate configuration without executing.
Use `--train` with a local model path and optional deps (`transformers`, `peft`, `torch`).
Sample datasets for dry-run checks live under `docs/training_sample.json`
and `docs/dpo_sample.json`.

Example `settings.json` (command-based training):
```json
{
  "plugins": {
    "configs": {
      "autocapture.builtin.training": {
        "pipelines": {
          "lora": {
            "command": ["python", "scripts/run_lora.py", "--dataset", "{dataset_path}"],
            "args": ["--out", "{output_dir}", "--params", "{params_json}"],
            "timeout_s": 3600
          }
        }
      }
    }
  }
}
```
Placeholders: `{dataset_path}`, `{output_dir}`, `{run_id}`, `{params_json}`.

## TNAS Service Hosting (Optional)

Autocapture local mode on Windows does **not** require Docker. Qdrant and FFmpeg
are bundled in release builds, and the app starts a local Qdrant sidecar
automatically when `qdrant.url` points at localhost. Use TNAS/Docker only when
you want centralized storage or shared services.

1. **Install Docker on the TNAS.** Use TerraMaster’s Docker Center (or SSH with
   `docker`/`docker-compose`) so Postgres, Qdrant, Prometheus, and Grafana all
   run on the NAS. Confirm the NAS has sufficient CPU/RAM headroom before
   onboarding the stack.

2. **Provision service directories.** In the NAS file manager, create a
   top-level share such as `autocapture` and add subfolders `captures`,
   `postgres`, `qdrant`, and `metrics`. These directories will be bind-mounted
   into containers to persist data on the NAS disks.

3. **Harden access.** Disable guest access, create a dedicated service account
   for the workstation, and grant it read/write rights to the share. Autocapture
   encrypts artifacts with AES-GCM before uploading, but access control still
   prevents tampering.

4. **Expose SMB for workstation sync.** Map the share to the Windows
   workstation so the capture pipeline can stream encrypted screenshots and
   retrieve logs if necessary. Use a persistent drive letter or the UNC path in
   scheduled scripts.

## Observability Stack

1. **Prometheus**
   - Deploy Prometheus as a NAS container mounting `metrics/prom-data` and a
     configuration file that scrapes the workstation exporter (e.g.,
     `<capture-host>:9005`).
   - Verify connectivity from the NAS to the workstation by running
     `docker exec prometheus wget -qO- http://<capture-host>:9005/metrics`.

2. **Grafana Dashboards**
   - Host Grafana on the NAS alongside Prometheus. After the container starts,
     log in at `http://<grafana-host>:3000`, add Prometheus (e.g.,
     `http://prometheus:9090`) as a data source, and import
     `docs/dashboard.json`.
   - Configure folders and permissions if you create additional dashboards.

3. **Alerting**
   - Extend the NAS-hosted `prometheus.yml` with alert rules such as:
     ```yaml
     groups:
        - name: autocapture
          rules:
            - alert: OCRBacklogHigh
              expr: ocr_backlog > 2000
              for: 15m
            - alert: StorageQuota
              expr: media_folder_size_gb > 2600
              for: 10m
      ```
   - Use Grafana Alerting, SMTP, or a webhook integration from the NAS to
     deliver notifications when metrics cross thresholds.

## Deployment Workflow

1. **Configuration** – Copy `config/example.yml` to the workstation, adjust
   paths, NAS hostnames, and the AES key provider details.
2. **Capture Service** – Install the capture orchestrator as a Windows Service
   (NSSM or `winsvc`) pointing at the interpreter inside your project virtual
   environment, for example:
   ```powershell
   nssm install Autocapture "C:\Path\To\repo\.venv\Scripts\python.exe" "-m" "autocapture.main" "--config" "C:/Path/To/autocapture.yml"
   ```
3. **NAS Containers** – On the TNAS, run the Docker Compose stack for Postgres,
   Qdrant, Prometheus, and Grafana, mounting the directories created above.
4. **GPU Drivers** – Keep workstation NVIDIA drivers/CUDA in sync with the OCR
   and embedding dependencies.
5. **Testing** – Run synthetic capture tests to verify deduplication, OCR
   throughput, and NAS connectivity before enabling full retention.
6. **Maintenance** – Schedule monthly audits: check NAS SMART stats, vacuum
   Postgres tables (via `docker exec` on the NAS), rotate logs, and validate
   Prometheus alerts.
