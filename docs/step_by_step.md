# Step-by-Step Deployment Checklist

This guide expands the high-level rollout plan into concrete tasks you can
follow sequentially. Treat each section as a gate: finish every checklist
item before moving to the next step so the capture pipeline stays performant
and observable.

> **Prerequisites**
>
> * Windows 11 workstation with administrator privileges, CUDA-capable GPU,
>   and Python 3.11+.
> * TNAS appliance accessible over SMB with enough capacity for encrypted long-term storage.
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
2. **Install Autocapture with Windows extras and tooling.**
   ```powershell
   python -m pip install -e .[windows]
   python -m pip install ruff black
   ```
   This pulls in `mss` and `psutil` for the Windows capture backend and sets up
   the linting stack used by the repository.
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
   * `ocr`, `embeddings`, `database`, `qdrant`, `encryption`, and
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
   orchestrator with [NSSM](https://nssm.cc/):
   ```powershell
   nssm install Autocapture "C:\Path\To\python.exe" "-m" "autocapture.main" "--config" "C:/Path/To/autocapture.yml" "--log-dir" "C:/Path/To/logs"
   nssm set Autocapture AppDirectory "C:/Path/To/repo"
   nssm start Autocapture
   ```
   This ensures the capture service survives reboots and starts before you log
   in.

---

## Step 2 – Prepare the TNAS share for long-term storage

1. **Create or repurpose a secure share.** Use the TNAS web console’s Storage
   Manager to create a dedicated share (e.g., `autocapture`). If your TNAS
   firmware supports encrypted volumes/folders, enable the option here so the
   disks remain protected without installing additional packages. Otherwise,
   rely on Autocapture’s built-in AES-GCM encryption and keep share access
   restricted to your workstation account.
2. **Lock down permissions.** Disable guest/anonymous access, require a strong
   username/password, and ensure the share is readable/writable only by your
   Windows user (and administrative break-glass accounts).
3. **Create directory structure for the services.** From the TNAS file manager
   or an SMB mount, add folders such as `captures`, `postgres`, `qdrant`, and
   `metrics` inside the share. These become bind mounts for Docker containers
   running on the workstation.
4. **Mount the share on Windows.** Map it to a drive letter (e.g., `Z:`) via
   File Explorer → “Map network drive,” or run:
   ```powershell
   New-PSDrive -Name "Z" -PSProvider FileSystem -Root "\\nas\autocapture" -Persist
   ```
   Confirm you can read/write files and that latency is acceptable for bulk
   uploads. The capture service will stream encrypted artifacts and database
   volumes to this location.

---

## Step 3 – Run infrastructure containers on the workstation

1. **Install Docker Desktop (or another Windows container runtime).** Enable
   WSL2 integration and grant the runtime access to the mapped NAS drive (`Z:`)
   in *Settings → Resources → File Sharing*.
2. **Create a local infrastructure folder** inside the repository (or another
   convenient path) with the following files:
   * `docker-compose.yml` – orchestrates Postgres, Qdrant, Prometheus, and
     Grafana.
   * `prometheus.yml` – scrape configuration for the capture metrics.
3. **Author the Compose file.** Example (`docker-compose.yml`):
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
           source: Z:/autocapture/postgres
           target: /var/lib/postgresql/data
       ports:
         - "5432:5432"

     qdrant:
       image: qdrant/qdrant:latest
       volumes:
         - type: bind
           source: Z:/autocapture/qdrant
           target: /qdrant/storage
       ports:
         - "6333:6333"

     prometheus:
       image: prom/prometheus:latest
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
         - type: bind
           source: Z:/autocapture/metrics/prom-data
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
           source: Z:/autocapture/metrics/grafana-data
           target: /var/lib/grafana
       ports:
         - "3000:3000"
   ```
   Adjust the `source:` paths if you mounted the NAS share to a different drive
   letter or directory.
4. **Configure Prometheus scraping.** Create `prometheus.yml` alongside the
   Compose file:
   ```yaml
   global:
     scrape_interval: 15s
   scrape_configs:
     - job_name: autocapture
       static_configs:
         - targets: ['host.docker.internal:9005']
   ```
   Replace `host.docker.internal` with the workstation hostname/IP if the
   runtime cannot resolve it.
5. **Launch the stack.** From PowerShell:
   ```powershell
   cd C:\Path\To\infrastructure
   docker compose up -d
   ```
   Verify each container is healthy via `docker compose ps`.
6. **Import dashboards and alerts.** Visit `http://localhost:3000`, log into
   Grafana, add Prometheus (`http://host.docker.internal:9090`) as a data
   source, and import `docs/dashboard.json`. Extend `prometheus.yml` with alert
   rules for OCR backlog, storage usage, and service heartbeats as needed.

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
           source: Z:/autocapture/postgres
           target: /var/lib/postgresql/data
       ports:
         - "5432:5432"

     qdrant:
       image: qdrant/qdrant:latest
       volumes:
         - type: bind
           source: Z:/autocapture/qdrant
           target: /qdrant/storage
       ports:
         - "6333:6333"
   ```
   Restart Docker Compose so all services come up.
2. **Update `autocapture.yml`** so the database, Qdrant, and Grafana entries
   point to `localhost` (matching the Compose stack) and confirm any paths that
   move long-term artifacts (e.g., `storage`) target the mapped NAS drive.
3. **Run the OCR worker.** From the workstation (with the virtual environment
   active), start the worker loop in a dedicated PowerShell window:
   ```powershell
   python -m autocapture.ocr.pipeline --config autocapture.yml
   ```
   (Use `Start-Process PowerShell -ArgumentList ...` if you want it detached.)
   Monitor the console and Prometheus metrics to confirm batches complete
   within the configured latency window.
4. **Schedule the embedding batcher.** Create a Task Scheduler entry that runs:
   ```powershell
   python -m autocapture.embeddings.pipeline --config C:\Path\To\autocapture.yml
   ```
   on the cron-like schedule defined in `embeddings.schedule_cron`. For manual
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
   * In the workstation Compose file, set `restart: unless-stopped` on
     Prometheus, Grafana, Postgres, and Qdrant so they recover automatically.
2. **Storage hygiene.**
   * Monitor the `autocapture_disk_usage_gb` metric. When the quota is reached,
     the retention worker prunes oldest images; still, set quarterly reminders
     to audit usage manually.
   * Periodically validate the NAS share’s protection—confirm the encrypted
     share remains locked down and that Autocapture’s AES-GCM encryption is
     still enabled in configuration.
3. **Database care.** Schedule monthly maintenance:
   ```bash
   docker exec -it postgres psql -U autocapture -d autocapture -c 'VACUUM;'
   docker exec -it postgres psql -U autocapture -d autocapture -c 'REINDEX;'
   ```
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
