"""Plugin error types."""

from __future__ import annotations


class PluginError(RuntimeError):
    """Base error for plugin system."""


class PluginManifestError(PluginError):
    """Invalid plugin manifest."""


class PluginDiscoveryError(PluginError):
    """Failed to discover plugin metadata."""


class PluginResolutionError(PluginError):
    """Failed to resolve an extension."""


class PluginPolicyError(PluginError):
    """Policy gate rejected a plugin extension."""


class PluginLockError(PluginError):
    """Plugin lock/hash mismatch."""
