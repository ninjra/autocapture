# Step-by-Step Deployment Checklist

This guide expands the high-level rollout plan into concrete tasks you can
follow sequentially. Treat each section as a gate: finish every checklist
item before moving to the next step so the capture pipeline stays performant
and observable.

> **Prerequisites**
>
> * Windows 11 workstation with administrator privileges, CUDA-capable GPU,
>   and Python 3.12.
> * TNAS appliance with Docker support and sufficient capacity for long-term storage.
> * Basic familiarity with PowerShell on Windows and a POSIX shell on the NAS.

---

## Step 1 – Prepare the workstation capture environment

1. **Create and activate a virtual environment.** Run the following from the
   project root inside PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```
2. **Install Autocapture (and optional dev tooling).**
   ```powershell
   python -m pip install -e .
   poetry install --with dev
   ```
   The runtime install is enough to run captures. Use Poetry if you need the
   dev stack (ruff/black/pytest/pyinstaller) that CI uses.
3. **Duplicate and customize the configuration.**
   ```powershell
   Copy-Item config/example.yml autocapture.yml
   ```
   Open `autocapture.yml` in your editor and adjust the following:
   * `capture.staging_dir` → fast local NVMe path that can absorb bursts before
     the NAS sync kicks in (e.g., `D:/autocapture/staging`).
   * `capture.hid.min_interval_ms` / `fps_soft_cap` → tune cadence.
   * `capture.hid.block_fullscreen` → leave `true` so fullscreen apps are
     ignored automatically.
   * `ocr`, `embed`, `database`, `qdrant`, `encryption`, and
     `observability` URLs → point at placeholders for now; you will update them
     again after the NAS stack is online.
4. **Create directories referenced in the config.** For example:
   ```powershell
   New-Item -ItemType Directory -Force -Path D:\autocapture\staging
   New-Item -ItemType Directory -Force -Path D:\autocapture\logs
   ```
5. **Launch the orchestrator in a console session.**
   ```powershell
   python -m autocapture.main --config autocapture.yml --log-dir D:/autocapture/logs
   ```
   You should see `Capture service started`. Move the mouse or type to confirm
   that files appear in the staging directory. Close the console with
   `Ctrl+C` when done testing.
6. **Review logs and metrics locally.** The orchestrator writes structured logs
   under the `--log-dir` you provided. You can also hit the Prometheus endpoint
   (default `http://localhost:9005/metrics`) with:
   ```powershell
   Invoke-WebRequest http://localhost:9005/metrics | Select-Object -ExpandProperty Content
   ```
   Confirm counters like `autocapture_captures_total` increment while you
   interact with the workstation.
7. **Optional – Install as a Windows Service.** Once satisfied, register the
   orchestrator with [NSSM](https://nssm.cc/) using the interpreter inside your
   virtual environment so child processes always inherit the same Python
   runtime:
   ```powershell
   nssm install Autocapture "C:\Path\To\repo\.venv\Scripts\python.exe" "-m" "autocapture.main" "--config" "C:/Path/To/autocapture.yml" "--log-dir" "C:/Path/To/logs"
   nssm set Autocapture AppDirectory "C:/Path/To/repo"
   nssm start Autocapture
   ```
   This ensures the capture service survives reboots and starts before you log
   in.

---

## Step 2 – Prepare the TNAS for container hosting and storage

1. **Enable Docker on the NAS.** Install TerraMaster’s Docker Center (or enable
   the CLI via SSH) so you can run Compose stacks directly on the appliance.
   Verify `docker ps` works from an SSH session before proceeding.
2. **Provision a dedicated share.** In the NAS UI, create a share such as
   `autocapture` and add subdirectories `captures`, `postgres`, `qdrant`, and
   `metrics`. These paths (e.g., `/volume1/autocapture/postgres`) become bind
   mounts for NAS-hosted containers and storage for encrypted artifacts.
3. **Harden access.** Disable guest access, create a service account with a
   strong passphrase, and restrict the share to that account plus administrators.
   Autocapture will AES-GCM encrypt screenshots before upload, but tight ACLs
   prevent accidental modification.
4. **Map the share to Windows.** Use File Explorer or PowerShell to mount the
   UNC path (e.g., `\\nas\autocapture`) so the capture service can stream
   encrypted files. A persistent drive letter simplifies scripts and log review.

---

## Step 3 – Run infrastructure containers on the TNAS

1. **Create an infrastructure workspace.** SSH into the NAS and create a
   directory (e.g., `/volume1/autocapture/infra`). Place the Compose file and
   Prometheus configuration here so they reside alongside the data folders.
2. **Author the Compose file (`docker-compose.yml`).** Example:
   ```yaml
   services:
     postgres:
       image: postgres:16
       environment:
         POSTGRES_DB: autocapture
         POSTGRES_USER: autocapture
         POSTGRES_PASSWORD: strong_password
       volumes:
         - type: bind
           source: /volume1/autocapture/postgres
           target: /var/lib/postgresql/data
       ports:
         - "5432:5432"

     qdrant:
       image: qdrant/qdrant:latest
       volumes:
         - type: bind
           source: /volume1/autocapture/qdrant
           target: /qdrant/storage
       ports:
         - "6333:6333"

     prometheus:
       image: prom/prometheus:latest
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
         - type: bind
           source: /volume1/autocapture/metrics/prom-data
           target: /prometheus
       ports:
         - "9090:9090"

     grafana:
       image: grafana/grafana:latest
       environment:
         GF_SECURITY_ADMIN_USER: admin
         GF_SECURITY_ADMIN_PASSWORD: change_me
       volumes:
         - type: bind
           source: /volume1/autocapture/metrics/grafana-data
           target: /var/lib/grafana
       ports:
         - "3000:3000"
   ```
   Adjust the `source:` paths if your NAS mounts volumes under a different
   prefix.
3. **Configure Prometheus scraping (`prometheus.yml`).**
   ```yaml
   global:
     scrape_interval: 15s
   scrape_configs:
     - job_name: autocapture
       static_configs:
         - targets: ['workstation.lan:9005']
   ```
   Replace `workstation.lan` with the capture workstation’s hostname or IP.
4. **Launch the stack.** From the NAS shell:
   ```bash
   cd /volume1/autocapture/infra
   docker compose up -d
   ```
   Confirm container health with `docker compose ps` and review logs if any
   service fails.
5. **Import dashboards and alerts.** Visit `http://nas.local:3000`, log into
   Grafana, add Prometheus (`http://prometheus:9090`) as a data source, and
   import `docs/dashboard.json`. Extend `prometheus.yml` with alert rules for OCR
   backlog, storage usage, and service heartbeats as needed.

---

## Step 4 – Enable OCR and embedding pipelines

1. **Ensure Postgres and Qdrant are defined in Compose.** If you split the stack
   across files, add services similar to the example below so both databases use
   the NAS-backed bind mounts:
   ```yaml
     postgres:
       image: postgres:16
       environment:
         POSTGRES_DB: autocapture
         POSTGRES_USER: autocapture
         POSTGRES_PASSWORD: strong_password
       volumes:
         - type: bind
           source: /volume1/autocapture/postgres
           target: /var/lib/postgresql/data
       ports:
         - "5432:5432"

     qdrant:
       image: qdrant/qdrant:latest
       volumes:
         - type: bind
           source: /volume1/autocapture/qdrant
           target: /qdrant/storage
       ports:
         - "6333:6333"
   ```
   Restart Docker Compose on the NAS so all services come up with the updated
   configuration.
2. **Update `autocapture.yml`** so the database, Qdrant, and Grafana entries
   point to your NAS hostnames (e.g., `nas.local`) and confirm any paths that
   move long-term artifacts (e.g., `storage`) target the mapped NAS share on
   Windows.
3. **Run the OCR worker.** From the workstation (with the virtual environment
   active), start the worker loop in a dedicated PowerShell window. When you
   detach it (e.g., via `Start-Process` or Task Scheduler), call the interpreter
   inside `.venv` explicitly so you do not fall back to the global Python
   installation:
   ```powershell
   .\.venv\Scripts\python.exe -m autocapture.ocr.pipeline --config autocapture.yml
   ```
   (Use `Start-Process PowerShell -ArgumentList ...` if you want it detached.)
   Monitor the console and Prometheus metrics to confirm batches complete
   within the configured latency window.
4. **Schedule the embedding batcher.** Create a Task Scheduler entry that runs:
   ```powershell
   C:\Path\To\repo\.venv\Scripts\python.exe -m autocapture.embeddings.pipeline --config C:\Path\To\autocapture.yml
   ```
   on the cron-like schedule defined in `embed.schedule_cron`. For manual
   execution, run the same command in a console to ensure vectors land in
   Qdrant (`autocapture_embedding_indexed_total` metric should rise).
5. **Verify data flow end-to-end.** Confirm captures transition from the staging
   folder to the NAS, OCR spans populate Postgres (`SELECT COUNT(*) FROM ocr_spans;`),
   and vectors appear in Qdrant via `qdrant_client` or the web UI.

---

## Step 5 – Establish ongoing maintenance

1. **Service hardening and resilience.**
   * Ensure the capture orchestrator, OCR worker, and embedding scheduler are
     configured to auto-start after reboots (Windows Service, scheduled task,
     or background PowerShell).
   * In the NAS Compose file, set `restart: unless-stopped` on
     Prometheus, Grafana, Postgres, and Qdrant so they recover automatically.
2. **Storage hygiene.**
   * Monitor the `media_folder_size_gb` metric. When the quota is reached,
      the retention worker prunes oldest images; still, set quarterly reminders
      to audit usage manually.
   * Periodically validate the NAS share’s protection—confirm permissions remain
     locked down and that Autocapture’s AES-GCM encryption is still enabled in
     configuration.
3. **Database care.** Schedule monthly maintenance:
   ```bash
   docker exec -it postgres psql -U autocapture -d autocapture -c 'VACUUM;'
   docker exec -it postgres psql -U autocapture -d autocapture -c 'REINDEX;'
   ```
   Run these commands from an SSH session on the NAS so they target the
   containerized Postgres instance.
   Rotate database credentials annually and update `autocapture.yml`.
4. **Alert review.** Test alert channels quarterly by temporarily raising a
   metric above threshold (e.g., stop the OCR worker) to ensure notifications
   still reach you.
5. **Software updates.** Keep NVIDIA drivers, CUDA libraries, and Python
   packages current. Test updates in a staging branch of the repo before
   deploying to your production environment.

Completing these steps will leave you with a capture pipeline that remains
fully local, GPU-accelerated, encrypted at rest, and observable from end to
end.
