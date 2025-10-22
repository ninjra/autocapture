# Operations Guide

## TNAS Service Hosting

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
     `workstation.lan:9005`).
   - Verify connectivity from the NAS to the workstation by running
     `docker exec prometheus wget -qO- http://workstation.lan:9005/metrics`.

2. **Grafana Dashboards**
   - Host Grafana on the NAS alongside Prometheus. After the container starts,
     log in at `http://nas.local:3000`, add Prometheus (e.g.,
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
             expr: autocapture_ocr_backlog > 2000
             for: 15m
           - alert: StorageQuota
             expr: autocapture_disk_usage_gb > 2600
             for: 10m
     ```
   - Use Grafana Alerting, SMTP, or a webhook integration from the NAS to
     deliver notifications when metrics cross thresholds.

## Deployment Workflow

1. **Configuration** – Copy `config/example.yml` to the workstation, adjust
   paths, NAS hostnames, and the AES key provider details.
2. **Capture Service** – Install the capture orchestrator as a Windows Service
   (NSSM or `winsvc`) invoking `python -m autocapture.main --config
   autocapture.yml`.
3. **NAS Containers** – On the TNAS, run the Docker Compose stack for Postgres,
   Qdrant, Prometheus, and Grafana, mounting the directories created above.
4. **GPU Drivers** – Keep workstation NVIDIA drivers/CUDA in sync with the OCR
   and embedding dependencies.
5. **Testing** – Run synthetic capture tests to verify deduplication, OCR
   throughput, and NAS connectivity before enabling full retention.
6. **Maintenance** – Schedule monthly audits: check NAS SMART stats, vacuum
   Postgres tables (via `docker exec` on the NAS), rotate logs, and validate
   Prometheus alerts.
