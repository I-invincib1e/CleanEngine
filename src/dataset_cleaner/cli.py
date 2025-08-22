#!/usr/bin/env python3
"""
CleanEngine CLI - Professional command-line interface for data cleaning and analysis
"""

import sys
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.dataset_cleaner.core.cleaner import DatasetCleaner
from src.dataset_cleaner.utils.config_manager import ConfigManager
from scripts.create_sample_data import create_sample_datasets
from scripts.run_tests import run_test_suite

# Initialize Typer app
app = typer.Typer(
    name="cleanengine",
    help="🧹 The Ultimate Data Cleaning & Analysis Toolkit",
    add_completion=False,
    rich_markup_mode="rich",
)

# Initialize Rich console
console = Console()

# Global configuration
config = ConfigManager()

def print_banner():
    """Display the CleanEngine banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🧹 CleanEngine 🧹                        ║
    ║              The Ultimate Data Cleaning Toolkit              ║
    ║                                                              ║
    ║  Transform messy datasets into clean, insights-rich data    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))

def validate_file_path(file_path: str) -> Path:
    """Validate and return file path"""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File '{file_path}' not found![/red]")
        raise typer.Exit(1)
    if not path.is_file():
        console.print(f"[red]Error: '{file_path}' is not a file![/red]")
        raise typer.Exit(1)
    return path

def get_supported_formats():
    """Return list of supported file formats"""
    return [".csv", ".xlsx", ".xls", ".json", ".parquet", ".feather"]

@app.command()
def clean(
    file_path: str = typer.Argument(..., help="Path to the dataset file to clean"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for cleaned data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output files")
):
    """Clean a dataset using CleanEngine's intelligent pipeline"""
    
    if verbose:
        print_banner()
    
    try:
        # Validate input file
        input_path = validate_file_path(file_path)
        
        # Set output path
        if output_path is None:
            output_path = f"Cleans-{input_path.stem}"
        
        output_path = Path(output_path)
        
        # Check if output exists
        if output_path.exists() and not force:
            if not Confirm.ask(f"Output directory '{output_path}' exists. Overwrite?"):
                raise typer.Exit(0)
        
        # Initialize cleaner
        cleaner = DatasetCleaner()
        
        # Start cleaning process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Cleaning dataset...", total=None)
            
            # Load data
            progress.update(task, description="Loading dataset...")
            df = cleaner.load_data(str(input_path))
            
            # Clean data
            progress.update(task, description="Applying cleaning pipeline...")
            cleaned_df = cleaner.clean_dataset(str(input_path))
            
            # Save results
            progress.update(task, description="Saving cleaned data and reports...")
            # Create output folder and save results
            output_folder = cleaner.create_output_folder(str(input_path))
            cleaner.save_results(cleaned_df, cleaner.report, str(output_folder))
            
            progress.update(task, description="Complete!", completed=True)
        
        # Display results summary
        console.print(f"\n[bold green]✅ Cleaning completed successfully![/bold green]")
        
        results_table = Table(title="Cleaning Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Input rows", str(len(df)))
        results_table.add_row("Output rows", str(len(cleaned_df)))
        results_table.add_row("Input columns", str(len(df.columns)))
        results_table.add_row("Output columns", str(len(cleaned_df.columns)))
        results_table.add_row("Output directory", str(output_folder))
        
        console.print(results_table)
        
    except Exception as e:
        console.print(f"[red]Error during cleaning: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def samples(
    output_dir: str = typer.Option("sample_data", "--output", "-o", help="Output directory for sample datasets"),
    count: int = typer.Option(3, "--count", "-n", help="Number of sample datasets to create"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Create sample datasets for testing and demonstration"""
    
    if verbose:
        print_banner()
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Creating sample datasets...", total=count)
            
            # Create sample datasets
            created_files = create_sample_datasets(output_path, count)
            
            for i, file_path in enumerate(created_files):
                progress.update(task, description=f"Created {file_path.name}...", completed=i+1)
        
        # Display results
        console.print(f"\n[bold green]✅ Created {len(created_files)} sample datasets![/bold green]")
        
        files_table = Table(title="Sample Datasets Created")
        files_table.add_column("File", style="cyan")
        files_table.add_column("Size", style="white")
        files_table.add_column("Description", style="white")
        
        for file_path in created_files:
            size = file_path.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
            
            descriptions = {
                "sample_clean.csv": "Clean dataset with no issues",
                "sample_mixed.csv": "Mixed data types with some issues",
                "sample_dirty.csv": "Dirty dataset with many problems"
            }
            
            desc = descriptions.get(file_path.name, "Sample dataset")
            files_table.add_row(file_path.name, size_str, desc)
        
        console.print(files_table)
        console.print(f"\n[bold]Output directory:[/bold] {output_path}")
        console.print(f"[bold]Try cleaning one:[/bold] cleanengine clean {output_path}/sample_mixed.csv")
        
    except Exception as e:
        console.print(f"[red]Error creating sample datasets: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    coverage: bool = typer.Option(False, "--coverage", help="Run tests with coverage report")
):
    """Run the CleanEngine test suite"""
    
    try:
        console.print("[bold]🧪 Running CleanEngine Test Suite...[/bold]")
        
        # Run tests
        test_results = run_test_suite(verbose=verbose, coverage=coverage)
        
        if test_results["success"]:
            console.print(f"\n[bold green]✅ All tests passed! ({test_results['total']} tests)[/bold green]")
        else:
            console.print(f"\n[bold red]❌ {test_results['failed']} tests failed![/bold red]")
            raise typer.Exit(1)
        
        # Show test summary
        summary_table = Table(title="Test Results Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total tests", str(test_results["total"]))
        summary_table.add_row("Passed", str(test_results["passed"]))
        summary_table.add_row("Failed", str(test_results["failed"]))
        
        console.print(summary_table)
        
    except Exception as e:
        console.print(f"[red]Error running tests: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def gui(
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit app"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host for Streamlit app")
):
    """Launch the CleanEngine Streamlit web interface"""
    
    print_banner()
    console.print(f"[bold]🌐 Launching CleanEngine Web Interface...[/bold]")
    console.print(f"[dim]URL: http://{host}:{port}[/dim]")
    
    try:
        from src.dataset_cleaner.interfaces.streamlit_app import launch_streamlit
        launch_streamlit(port=port, host=host)
    except Exception as e:
        console.print(f"[red]Error launching GUI: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def info():
    """Show detailed information about CleanEngine"""
    
    print_banner()
    
    # Show system information
    info_table = Table(title="System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Version", style="white")
    
    info_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    info_table.add_row("Platform", sys.platform)
    
    # Show package information
    try:
        import pandas as pd
        info_table.add_row("pandas", pd.__version__)
    except ImportError:
        info_table.add_row("pandas", "Not installed")
    
    try:
        import numpy as np
        info_table.add_row("numpy", np.__version__)
    except ImportError:
        info_table.add_row("numpy", "Not installed")
    
    console.print(info_table)
    
    # Show supported formats
    formats = get_supported_formats()
    console.print(f"\n[bold]Supported formats:[/bold] {', '.join(formats)}")

def main():
    """Main CLI entry point"""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    main()
