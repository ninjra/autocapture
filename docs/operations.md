# Operations Guide

## TNAS Share Hardening

1. **Use the TNAS UI for encryption when possible.** Many TerraMaster releases
   provide encrypted shared folders via *Control Panel → Shared Folder → Create
   → Encrypt*. Prefer this option because it does not require additional
   packages. Record the encryption password in a secure manager and test the
   unlock workflow after a reboot.

2. **Restrict access.** Whether the share is encrypted or not, disable guest
   access, require a unique service account with a strong passphrase, and
   assign read/write permissions only to that account and administrators.

3. **Create a dedicated directory structure.** Inside the share, add
   subfolders such as `captures`, `postgres`, `qdrant`, and `metrics`. Keep the
   layout stable so Docker bind mounts on the workstation remain valid even if
   the NAS reboots.

4. **Monitor the share state.** Periodically review the TNAS UI to confirm the
   folder remains encrypted (if enabled) and that no unauthorized users were
   granted access. Because Autocapture also encrypts artifacts before writing
   them, the combination ensures privacy even if full disk encryption is not
   available.

## Observability Stack

1. **Prometheus**
   - Run Prometheus inside Docker Desktop on the workstation using the
     `docker-compose.yml` described in the deployment checklist. Ensure the
     compose file mounts the NAS share for durable metrics storage and points to
     a `prometheus.yml` that scrapes `host.docker.internal:9005` (or your
     workstation IP).

2. **Grafana Dashboards**
   - Access Grafana at `http://localhost:3000`, authenticate with the admin
     credentials defined in Compose, and add Prometheus
     (`http://host.docker.internal:9090`) as a data source.
   - Import `docs/dashboard.json` to visualize capture throughput, OCR latency,
     GPU utilization, and NAS usage trends.

3. **Alerting**
   - Extend `prometheus.yml` with alert rules similar to:
     ```yaml
     groups:
       - name: autocapture
         rules:
           - alert: OCRBacklogHigh
             expr: autocapture_ocr_backlog > 2000
             for: 15m
           - alert: StorageQuota
             expr: autocapture_disk_usage_gb > 2600
             for: 10m
     ```
   - Route alerts through Grafana Alerting, SMTP, or another integration so you
     are notified when the pipeline falls behind.

## Deployment Workflow

1. **Configuration** – Copy `config/example.yml` to the workstation, adjust paths, database URLs, and encryption key provider.
2. **Services** – Install the capture service as a Windows Service (using `winsvc` or NSSM) pointing to `python -m autocapture.main --config autocapture.yml`.
3. **Docker Stack** – Run the Compose stack (Postgres, Qdrant, Prometheus, Grafana) on the workstation with bind mounts pointing to the mapped NAS share.
4. **GPU Drivers** – Keep NVIDIA drivers + CUDA toolkit aligned with PaddleOCR/EasyOCR and SentenceTransformer versions.
5. **Testing** – Run synthetic capture tests with the staging pipeline before enabling full retention to verify dedupe, OCR throughput, and metric coverage.
6. **Maintenance** – Schedule monthly audits: check NAS SMART stats, vacuum Postgres tables, rotate logs, and validate Prometheus alerts.
