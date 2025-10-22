# Step-by-Step Deployment Checklist

This guide expands the high-level rollout plan into concrete tasks you can
follow sequentially. Treat each section as a gate: finish every checklist
item before moving to the next step so the capture pipeline stays performant
and observable.

> **Prerequisites**
>
> * Windows 11 workstation with administrator privileges, CUDA-capable GPU,
>   and Python 3.11+.
> * TNAS appliance with SSH access and room to provision an encrypted volume.
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

## Step 2 – Provision the encrypted TNAS share

1. **Enable SSH and confirm sudo access.** From the TNAS web UI, enable SSH
   (Control Panel → Network Services). Log in via `ssh admin@<nas>` and run
   `sudo -v` to verify privileges.
2. **Identify the storage device.**
   ```bash
   lsblk
   df -h | grep -v tmpfs
   ```
   Note the block device backing your data volume (e.g., `/dev/md0`).
3. **Install cryptsetup if necessary.**
   ```bash
   sudo apt-get update && sudo apt-get install -y cryptsetup
   ```
4. **Un-mount the target device and initialize LUKS.**
   ```bash
   sudo umount /dev/md0
   sudo cryptsetup luksFormat /dev/md0
   sudo cryptsetup luksOpen /dev/md0 secure_pool
   ```
   Supply a strong passphrase. Record it in your password manager.
5. **Format and mount the encrypted mapper device.**
   ```bash
   sudo mkfs.ext4 /dev/mapper/secure_pool
   sudo mkdir -p /mnt/secure_pool
   sudo mount /dev/mapper/secure_pool /mnt/secure_pool
   ```
6. **Persist the configuration across reboots.**
   * Add an entry to `/etc/crypttab`:
     ```text
     secure_pool /dev/md0 none luks
     ```
   * Add a corresponding entry to `/etc/fstab`:
     ```text
     /dev/mapper/secure_pool /mnt/secure_pool ext4 defaults 0 2
     ```
   * Optionally create a keyfile (`dd if=/dev/urandom of=/root/.keys/secure_pool.key bs=32 count=1`)
     and reference it in `crypttab` for unattended unlocks.
7. **Create the Autocapture directory structure.**
   ```bash
   sudo mkdir -p /mnt/secure_pool/autocapture/{postgres,qdrant,captures,metrics}
   sudo chown -R admin:admin /mnt/secure_pool/autocapture
   ```
8. **Expose an SMB share.** In the TNAS UI, create a share named `autocapture`
   pointing at `/mnt/secure_pool/autocapture`. Restrict access to your Windows
   account and disable guest access.
9. **Mount the share on Windows.** Map it as a network drive (e.g., `Z:`) via
   File Explorer or `New-PSDrive`. This drive will hold database volumes,
   encrypted screenshots, and Prometheus data once the stack is online.

---

## Step 3 – Deploy Prometheus and Grafana on the NAS

1. **Create a Docker Compose file on the encrypted share** (`/mnt/secure_pool/autocapture/docker-compose.yml`):
   ```yaml
   services:
     prometheus:
       image: prom/prometheus:latest
       container_name: prometheus
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
         - ./prom-data:/prometheus
       ports:
         - "9090:9090"
     grafana:
       image: grafana/grafana:latest
       container_name: grafana
       environment:
         - GF_SECURITY_ADMIN_USER=admin
         - GF_SECURITY_ADMIN_PASSWORD=change_me
       volumes:
         - ./grafana-data:/var/lib/grafana
       ports:
         - "3000:3000"
   ```
2. **Author the Prometheus scrape configuration** (`prometheus.yml`):
   ```yaml
   global:
     scrape_interval: 15s
   scrape_configs:
     - job_name: autocapture
       static_configs:
         - targets: ['workstation.lan:9005']
   ```
   Replace `workstation.lan` with the hostname or IP of your Windows PC.
3. **Launch the stack.**
   ```bash
   cd /mnt/secure_pool/autocapture
   docker compose up -d
   ```
4. **Import the dashboard.** Browse to `http://<nas>:3000`, log in, add a data
   source pointing at `http://prometheus:9090`, and import `docs/dashboard.json`
   from the repository. Verify panels populate once the capture service emits
   metrics.
5. **Configure alerting.** Extend `prometheus.yml` with alert rules and start
   the Alertmanager of your choice (Grafana Alerting, SMTP, etc.). Suggested
   alerts:
   * OCR backlog over 2,000 jobs for 15 minutes.
   * Disk usage above 2.6 TB.
   * Capture service heartbeat missing for more than 2 scrape intervals.

---

## Step 4 – Enable OCR and embedding pipelines

1. **Deploy Postgres and Qdrant alongside Prometheus.** Extend the same Compose
   stack (or a separate one) with services:
   ```yaml
     postgres:
       image: postgres:16
       environment:
         - POSTGRES_DB=autocapture
         - POSTGRES_USER=autocapture
         - POSTGRES_PASSWORD=strong_password
       volumes:
         - ./postgres:/var/lib/postgresql/data
       ports:
         - "5432:5432"

     qdrant:
       image: qdrant/qdrant:latest
       volumes:
         - ./qdrant:/qdrant/storage
       ports:
         - "6333:6333"
   ```
   Restart Docker Compose so all services come up.
2. **Update `autocapture.yml`** on the workstation with the actual NAS URLs and
   credentials (`database.url`, `qdrant.url`, `observability.grafana_url`).
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
   * On the NAS, enable Docker restart policies (`restart: unless-stopped`) for
     Prometheus, Grafana, Postgres, and Qdrant.
2. **Storage hygiene.**
   * Monitor the `autocapture_disk_usage_gb` metric. When the quota is reached,
     the retention worker prunes oldest images; still, set quarterly reminders
     to audit usage manually.
   * Periodically validate the encryption state by checking `lsblk -f` to confirm
     the underlying device remains LUKS-encrypted.
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
