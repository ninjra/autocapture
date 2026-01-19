# Entry-Point Plugin Example

This folder is a minimal, offline-friendly entry-point plugin package example.
It is not installed by default.

Layout:
- `pyproject.toml` declares the `autocapture.plugins` entry point.
- `autocapture_plugins/example.entrypoint.yaml` is the manifest AutoCapture discovers.
- `autocapture_example_plugin/factories.py` implements the factory.

Install the package into the same environment that runs AutoCapture, then enable it:
```bash
poetry run autocapture plugins enable example.entrypoint --accept-hashes
```

The provider returns a stub answer and is meant for demos only.
