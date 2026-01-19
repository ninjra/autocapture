# Folder Plugin Template

This folder is a minimal, offline-friendly plugin example. Drop a copy into
`${capture.data_dir}/plugins/<your-plugin>` to test discovery.

1. Edit `plugin.yaml` with your plugin ID and extensions.
2. Implement factories in `plugin_module.py`.
3. Enable the plugin:
   ```bash
   poetry run autocapture plugins enable <plugin_id> --accept-hashes
   ```

Disabled plugins are discovered without importing code.
