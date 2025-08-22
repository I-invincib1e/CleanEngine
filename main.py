#!/usr/bin/env python3
"""
Project entrypoint with a friendly CLI.

Usage:
  python main.py clean <file> [out]    # run dataset cleaner (alias: c)
  python main.py tests                 # run unit tests (alias: t)
  python main.py samples               # create sample datasets (alias: s)
  python main.py gui                   # launch Streamlit app (alias: g)

Short flags (no subcommand):
  python main.py -c <file> [out]
  python main.py -t
  python main.py -s
  python main.py -g

If run without args, shows an interactive menu.
"""
import sys
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    import questionary
except Exception:
    questionary = None

app = typer.Typer(add_completion=False)
console = Console()

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def run(cmd: list[str]) -> int:
    console.print(f"[bold cyan]$ {' '.join(cmd)}[/bold cyan]")
    return subprocess.call(cmd)


# --- Commands (long) ---
@app.command()
def tests() -> int:
    """Run unit tests"""
    return run([sys.executable, str(PROJECT_ROOT / 'scripts' / 'run_tests.py')])


@app.command()
def samples() -> int:
    """Create sample datasets in project root"""
    return run([sys.executable, str(PROJECT_ROOT / 'scripts' / 'create_sample_data.py')])


@app.command()
def clean(input: str = typer.Argument(..., help="Input CSV/Excel file"),
          output: Optional[str] = typer.Argument(None, help="Optional output file name")) -> int:
    """Clean a dataset and produce organized outputs."""
    cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'run_cleaner.py'), input]
    if output:
        cmd.append(output)
    return run(cmd)


@app.command()
def gui() -> int:
    """Launch the Streamlit GUI."""
    return run(['streamlit', 'run', str(PROJECT_ROOT / 'src' / 'dataset_cleaner' / 'interfaces' / 'streamlit_app.py')])


# --- Short command aliases ---
@app.command('t')
def t_alias() -> int:
    return tests()


@app.command('s')
def s_alias() -> int:
    return samples()


@app.command('c')
def c_alias(input: str = typer.Argument(...), output: Optional[str] = typer.Argument(None)) -> int:
    return clean(input, output)


@app.command('g')
def g_alias() -> int:
    return gui()


# --- Interactive menu ---

def show_menu() -> int:
    console.print(Panel("Automated Dataset Cleaner", subtitle="Beautiful CLI", style="green"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Action"), table.add_column("Description"), table.add_column("Command")
    table.add_row("Create samples", "Generate example CSVs", "python main.py s")
    table.add_row("Clean a file", "Select and clean a dataset", "python main.py c <file>")
    table.add_row("Run tests", "Execute unit tests", "python main.py t")
    table.add_row("Launch GUI", "Open Streamlit interface", "python main.py g")
    console.print(table)

    if questionary is None:
        console.print("[yellow]questionary not installed; showing command equivalents:[/yellow]")
        console.print("- python main.py s")
        console.print("- python main.py c <file>")
        console.print("- python main.py t")
        console.print("- python main.py g")
        return 0

    choice = questionary.select(
        "What would you like to do?",
        choices=["Create samples", "Clean a file", "Run tests", "Launch GUI", "Exit"],
    ).ask()

    if choice == "Create samples":
        return samples()
    if choice == "Clean a file":
        file_path = questionary.path("Select a CSV/Excel file:").ask()
        if not file_path:
            return 1
        return clean(file_path)
    if choice == "Run tests":
        return tests()
    if choice == "Launch GUI":
        return gui()
    return 0


# --- Global short flags ---
@app.callback(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def global_shortcuts(
    c: Optional[str] = typer.Option(None, "-c", help="Clean: input file path"),
    t: bool = typer.Option(False, "-t", help="Run tests"),
    s: bool = typer.Option(False, "-s", help="Create samples"),
    g: bool = typer.Option(False, "-g", help="Launch GUI"),
):
    # Triggered before subcommand dispatch. If a short flag is used, execute immediately and exit.
    if c:
        raise typer.Exit(code=clean(c, None))
    if t:
        raise typer.Exit(code=tests())
    if s:
        raise typer.Exit(code=samples())
    if g:
        raise typer.Exit(code=gui())


def main() -> int:
    # If no arguments given, show interactive menu
    if len(sys.argv) == 1:
        return show_menu()
    app()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
