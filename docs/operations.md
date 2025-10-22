# Operations Guide

## TNAS Drive-Level Encryption

1. **Enable SSH and Admin Access**
   - Log in to the TNAS web console.
   - Navigate to **Control Panel → General Settings → Network Services** and enable SSH.
   - Ensure your admin account has sudo privileges.

2. **Identify the Storage Pool**
   - SSH into the NAS: `ssh admin@nas`.
   - Run `df -h` and `lsblk` to locate the volume that backs your Docker and data shares (e.g., `/dev/md0`).

3. **Create an Encrypted LUKS Container**
   - Install cryptsetup if absent: `sudo tnas-intall cryptsetup` (package names vary; TNAS uses a Debian derivative).
   - Unmount the target volume temporarily: `sudo umount /dev/md0`.
   - Initialize LUKS: `sudo cryptsetup luksFormat /dev/md0`.
   - Open the encrypted volume: `sudo cryptsetup luksOpen /dev/md0 secure_pool`.

4. **Format and Mount**
   - Format with EXT4: `sudo mkfs.ext4 /dev/mapper/secure_pool`.
   - Create a mount point: `sudo mkdir -p /mnt/secure_pool`.
   - Mount: `sudo mount /dev/mapper/secure_pool /mnt/secure_pool`.
   - Update `/etc/fstab` and `/etc/crypttab` to auto-unlock on boot. Store the keyfile on an encrypted USB or use a passphrase prompt during boot.

5. **Share the Encrypted Volume**
   - In the TNAS UI, create an SMB share pointing to `/mnt/secure_pool/autocapture`.
   - Restrict access to your workstation account and disable guest access.

6. **Key Management Best Practices**
   - Store the LUKS passphrase in a hardware password manager or sealed envelope.
   - Export a recovery key with `sudo cryptsetup luksAddKey /dev/md0` and keep it offline.
   - Rotate the passphrase annually and update the Windows Credential Manager entry used by the application.

## Observability Stack

1. **Prometheus**
   - Deploy the official Prometheus Docker image on the TNAS: `docker run -d --name prometheus -p 9090:9090 -v /secure_pool/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus`.
   - Add a job to scrape the workstation: 
     ```yaml
     scrape_configs:
       - job_name: autocapture
         static_configs:
           - targets: ['workstation.local:9005']
     ```

2. **Grafana Dashboards**
   - Launch Grafana: `docker run -d --name grafana -p 3000:3000 -v /secure_pool/grafana:/var/lib/grafana grafana/grafana`.
   - Import the provided dashboard (see `docs/dashboard.json`, to be generated once metrics flow).
   - Key panels: capture throughput, OCR latency (queue length / batch time), GPU utilization, NAS usage.

3. **Alerting**
   - Configure Prometheus alert rules:
     ```yaml
     groups:
       - name: autocapture
         rules:
           - alert: OCRBacklogHigh
             expr: autocapture_ocr_backlog > 2000
             for: 15m
             labels:
               severity: warning
             annotations:
               summary: OCR backlog exceeded 2k jobs
           - alert: StorageQuota
             expr: autocapture_disk_usage_gb > 2600
             for: 10m
             labels:
               severity: critical
             annotations:
               summary: NAS storage above 2.6 TB
     ```
   - Wire alerts to Grafana Alerting or an SMTP/Teams webhook.

## Deployment Workflow

1. **Configuration** – Copy `config/example.yml` to the workstation, adjust paths, database URLs, and encryption key provider.
2. **Services** – Install the capture service as a Windows Service (using `winsvc` or NSSM) pointing to `python -m autocapture.main --config autocapture.yml`.
3. **Docker Stack** – Deploy Postgres and Qdrant via `docker-compose` on the TNAS. Use bind mounts to the encrypted volume.
4. **GPU Drivers** – Keep NVIDIA drivers + CUDA toolkit aligned with PaddleOCR/EasyOCR and SentenceTransformer versions.
5. **Testing** – Run synthetic capture tests with the staging pipeline before enabling full retention to verify dedupe, OCR throughput, and metric coverage.
6. **Maintenance** – Schedule monthly audits: check NAS SMART stats, vacuum Postgres tables, rotate logs, and validate Prometheus alerts.
