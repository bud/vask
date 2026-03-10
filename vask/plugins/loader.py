"""Plugin system — discover, load, and manage tool plugins."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vask.logging import get_logger
from vask.tools.base import ToolProvider
from vask.tools.registry import ToolRegistry

logger = get_logger("plugins")


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Parsed plugin metadata from plugin.yaml."""

    name: str
    version: str
    description: str
    author: str = ""
    tools: list[str] = field(default_factory=list)
    required_env_vars: list[str] = field(default_factory=list)
    optional_env_vars: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedPlugin:
    """A plugin that has been loaded into memory."""

    manifest: PluginManifest
    tools: list[ToolProvider]
    path: Path
    enabled: bool = True
    errors: list[str] = field(default_factory=list)


class PluginLoader:
    """Discovers and loads plugins from the plugins directory and external paths."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._plugins: dict[str, LoadedPlugin] = {}
        self._plugin_dirs: list[Path] = [
            Path(__file__).parent,  # built-in plugins: vask/plugins/
        ]
        # Add user plugin directory
        user_plugin_dir = os.environ.get("VASK_PLUGIN_DIR", "")
        if user_plugin_dir:
            self._plugin_dirs.append(Path(user_plugin_dir).expanduser())

    def discover(self) -> list[PluginManifest]:
        """Discover all plugins without loading them."""
        manifests: list[PluginManifest] = []
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.is_dir():
                continue
            for child in sorted(plugin_dir.iterdir()):
                if not child.is_dir() or child.name.startswith("_"):
                    continue
                manifest_path = child / "plugin.yaml"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = self._parse_manifest(manifest_path)
                    manifests.append(manifest)
                except Exception as e:
                    logger.warning(f"Failed to parse manifest for {child.name}: {e}")
        return manifests

    def load_all(self, enabled_plugins: list[str] | None = None) -> dict[str, LoadedPlugin]:
        """Load all discovered plugins (or a subset)."""
        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.is_dir():
                continue
            for child in sorted(plugin_dir.iterdir()):
                if not child.is_dir() or child.name.startswith("_"):
                    continue
                manifest_path = child / "plugin.yaml"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = self._parse_manifest(manifest_path)
                except Exception as e:
                    logger.warning(f"Skipping {child.name}: bad manifest: {e}")
                    continue

                if enabled_plugins is not None and manifest.name not in enabled_plugins:
                    logger.debug(f"Skipping disabled plugin: {manifest.name}")
                    continue

                loaded = self._load_plugin(child, manifest)
                self._plugins[manifest.name] = loaded

                if loaded.enabled and not loaded.errors:
                    for tool in loaded.tools:
                        self._tool_registry.register(tool)
                    logger.info(
                        f"Loaded plugin '{manifest.name}' v{manifest.version} "
                        f"with {len(loaded.tools)} tools"
                    )
                elif loaded.errors:
                    logger.warning(
                        f"Plugin '{manifest.name}' loaded with errors: {loaded.errors}"
                    )

        return self._plugins

    def _parse_manifest(self, path: Path) -> PluginManifest:
        """Parse a plugin.yaml manifest file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return PluginManifest(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tools=data.get("tools", []),
            required_env_vars=data.get("required_env_vars", []),
            optional_env_vars=data.get("optional_env_vars", []),
            permissions=data.get("permissions", []),
            config_schema=data.get("config_schema", {}),
        )

    def _load_plugin(self, plugin_dir: Path, manifest: PluginManifest) -> LoadedPlugin:
        """Load a plugin's tools from its Python module."""
        errors: list[str] = []
        tools: list[ToolProvider] = []

        # Check required env vars
        for env_var in manifest.required_env_vars:
            if not os.environ.get(env_var):
                errors.append(f"Missing required env var: {env_var}")

        # Import the plugin module
        try:
            # Ensure the parent dir is in sys.path for external plugins
            parent = str(plugin_dir.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

            # Try importing as a vask.plugins.X module first (built-in)
            module_name = f"vask.plugins.{plugin_dir.name}"
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                # Fall back to importing by directory name (external)
                module = importlib.import_module(plugin_dir.name)

            # Get tools from the module's TOOLS list
            plugin_tools = getattr(module, "TOOLS", [])
            for tool_cls in plugin_tools:
                if isinstance(tool_cls, type):
                    # It's a class, instantiate it
                    tool = tool_cls()
                else:
                    # It's already an instance
                    tool = tool_cls
                if isinstance(tool, ToolProvider):
                    tools.append(tool)
                else:
                    errors.append(f"Tool {tool} does not implement ToolProvider protocol")

        except Exception as e:
            errors.append(f"Failed to import plugin module: {e}")

        return LoadedPlugin(
            manifest=manifest,
            tools=tools,
            path=plugin_dir,
            enabled=len([e for e in errors if "Missing required" in e]) == 0,
            errors=errors,
        )

    def get_plugin(self, name: str) -> LoadedPlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all loaded plugins with status."""
        result = []
        for name, plugin in self._plugins.items():
            result.append({
                "name": name,
                "version": plugin.manifest.version,
                "description": plugin.manifest.description,
                "enabled": plugin.enabled,
                "tools": [t.name for t in plugin.tools],
                "errors": plugin.errors,
            })
        return result

    def enable_plugin(self, name: str) -> bool:
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        plugin.enabled = True
        for tool in plugin.tools:
            self._tool_registry.register(tool)
        return True

    def disable_plugin(self, name: str) -> bool:
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        plugin.enabled = False
        for tool in plugin.tools:
            self._tool_registry.unregister(tool.name)
        return True
