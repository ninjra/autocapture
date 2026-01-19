# Entry-Point Plugin Template

Entry-point plugins are distributed as Python packages and discovered via the
`autocapture.plugins` entry point group.

## Packaging Layout
Include a manifest at:
```
autocapture_plugins/<plugin_id>.yaml
```
Optional assets can live under:
```
autocapture_plugins/<plugin_id>/assets/
```

## Example Package
See `docs/plugin-authoring/entrypoint-example/` for a complete, minimal package.

## `pyproject.toml` Example
```toml
[project.entry-points."autocapture.plugins"]
example.entrypoint = "example_plugin:factory"
```

The entry point name must match the manifest filename (`example.entrypoint` above).
Factories referenced in the manifest are only imported when the extension is used.
