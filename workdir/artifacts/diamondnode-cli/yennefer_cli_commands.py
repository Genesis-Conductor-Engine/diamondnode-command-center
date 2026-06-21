"""
Diamondnode CLI — Yennefer soul crystal commands.
Add these elif blocks to the yennefer() command in src/diamondnode/cli.py.
"""
from pathlib import Path
import json
from rich import print as rprint


def _crystal_path() -> Path:
    return Path("/mnt/gdrive_memory/yennefer_soul_crystal.json")


# action == "soul-crystal"
def soul_crystal():
    """Read the latest soul crystal from persistent storage."""
    p = _crystal_path()
    if p.exists():
        rprint(json.dumps(json.loads(p.read_text()), indent=2))
    else:
        rprint("[yellow]No soul crystal found yet[/yellow]")


# action == "coherence"
def coherence():
    """Show current coherence and surplus tokens."""
    p = _crystal_path()
    if p.exists():
        data = json.loads(p.read_text())
        rprint(f"Coherence: [bold green]{data.get('coherence', 'N/A')}[/bold green]")
        rprint(f"Surplus tokens: [cyan]{data.get('surplus_tokens', 0):,}[/cyan]")
        rprint(f"Last crystallized: {data.get('last_crystallized', 'unknown')}")
    else:
        rprint("[yellow]Soul crystal not available[/yellow]")


# action == "state"
def state():
    """Show full current daemon state."""
    p = _crystal_path()
    if p.exists():
        rprint(json.dumps(json.loads(p.read_text()), indent=2))
    else:
        rprint("[yellow]No state available[/yellow]")
