"""Typer CLI — record, transcribe, ask, mcp serve."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from vask.config import ensure_config
from vask.core.registry import register_defaults, registry

app = typer.Typer(
    name="vask",
    help="Vask — Voice-to-Action framework. Capture audio, transcribe, query LLMs.",
    no_args_is_help=True,
)
console = Console()

_initialized = False


def _init() -> None:
    global _initialized
    if not _initialized:
        register_defaults()
        _initialized = True


@app.command()
def record(
    input: str = typer.Option(None, "--input", "-i", help="Input source (mic, file, clipboard)"),
    llm: str = typer.Option(None, "--llm", "-l", help="LLM provider (gemini, claude, openai, openrouter)"),
    output: str = typer.Option(None, "--output", "-o", help="Output renderer (terminal, clipboard, json)"),
    system_prompt: str = typer.Option(None, "--system", "-s", help="System prompt for the LLM"),
    file: str = typer.Option(None, "--file", "-f", help="Audio file path (sets input to 'file')"),
) -> None:
    """Record audio, transcribe, send to LLM, and display response."""
    _init()
    config = ensure_config()

    from vask.core.pipeline import Pipeline

    pipeline = Pipeline(config)

    if file:
        input = "file"
        if "file" not in config.providers:
            from vask.config import ProviderConfig

            config.providers["file"] = ProviderConfig(name="file", type="input", extra={"path": file})
        else:
            config.providers["file"].extra["path"] = file

    asyncio.run(pipeline.record(input_name=input, llm_name=llm, output_name=output, system_prompt=system_prompt))


@app.command()
def transcribe(
    input: str = typer.Option(None, "--input", "-i", help="Input source"),
    output: str = typer.Option(None, "--output", "-o", help="Output renderer"),
    file: str = typer.Option(None, "--file", "-f", help="Audio file path"),
) -> None:
    """Record and transcribe only (no LLM)."""
    _init()
    config = ensure_config()

    from vask.core.pipeline import Pipeline

    pipeline = Pipeline(config)

    if file:
        input = "file"
        if "file" not in config.providers:
            from vask.config import ProviderConfig

            config.providers["file"] = ProviderConfig(name="file", type="input", extra={"path": file})
        else:
            config.providers["file"].extra["path"] = file

    text = asyncio.run(pipeline.transcribe(input_name=input, output_name=output))
    if output is None:
        pass  # already rendered by pipeline


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question or prompt text"),
    llm: str = typer.Option(None, "--llm", "-l", help="LLM provider"),
    output: str = typer.Option(None, "--output", "-o", help="Output renderer"),
    system_prompt: str = typer.Option(None, "--system", "-s", help="System prompt"),
) -> None:
    """Send a text query to an LLM (no audio)."""
    _init()
    config = ensure_config()

    from vask.core.pipeline import Pipeline

    pipeline = Pipeline(config)
    asyncio.run(pipeline.ask(question, llm_name=llm, output_name=output, system_prompt=system_prompt))


@app.command()
def providers() -> None:
    """List available providers."""
    _init()
    for category in ["input", "transcription", "llm", "output", "tool"]:
        names = registry.list_providers(category)
        console.print(f"[bold]{category}[/bold]: {', '.join(names)}")


@app.command()
def config() -> None:
    """Show config file location and current defaults."""
    from vask.config import CONFIG_FILE

    cfg = ensure_config()
    console.print(f"[bold]Config file:[/bold] {CONFIG_FILE}")
    console.print("[bold]Defaults:[/bold]")
    for k, v in cfg.defaults.items():
        console.print(f"  {k} = {v}")
    console.print("[bold]Providers:[/bold]")
    for name, p in cfg.providers.items():
        key_status = "set" if p.api_key else "missing"
        console.print(f"  {name} ({p.type}) — key: {key_status}, model: {p.model or 'default'}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind address"),
    port: int = typer.Option(8420, "--port", "-p", help="Port number"),
    log_json: bool = typer.Option(False, "--log-json", help="Output structured JSON logs"),
) -> None:
    """Start the Vask REST API server."""
    _init()
    cfg = ensure_config()

    from vask.logging import setup_logging

    setup_logging(level="INFO", json_output=log_json)

    from vask.api.server import run_server

    run_server(cfg, host=host, port=port)


@app.command()
def plugins() -> None:
    """List available plugins and their status."""
    _init()
    cfg = ensure_config()

    from vask.plugins.loader import PluginLoader
    from vask.tools.registry import ToolRegistry

    tool_reg = ToolRegistry()
    loader = PluginLoader(tool_reg)
    loaded = loader.load_all()

    if not loaded:
        console.print("[dim]No plugins found.[/dim]")
        return

    for name, plugin in loaded.items():
        status = "[green]enabled[/green]" if plugin.enabled else "[red]disabled[/red]"
        console.print(f"[bold]{name}[/bold] v{plugin.manifest.version} — {status}")
        console.print(f"  {plugin.manifest.description}")
        if plugin.tools:
            tool_names = [t.name for t in plugin.tools]
            console.print(f"  Tools: {', '.join(tool_names)}")
        if plugin.errors:
            for err in plugin.errors:
                console.print(f"  [yellow]⚠ {err}[/yellow]")


@app.command(name="mcp")
def mcp_serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse"),
) -> None:
    """Start Vask as an MCP server."""
    _init()
    cfg = ensure_config()

    from vask.mcp.server import run_server

    run_server(cfg, transport=transport)
