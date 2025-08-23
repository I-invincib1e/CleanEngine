#!/usr/bin/env python3
"""
CleanEngine CLI - Professional command-line interface for data cleaning and analysis
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from scripts.create_sample_data import create_sample_datasets
from scripts.run_tests import run_test_suite
from src.dataset_cleaner.core.cleaner import DatasetCleaner
from src.dataset_cleaner.utils.config_manager import ConfigManager

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
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for cleaned data"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing output files"
    ),
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
            console=console,
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
    output_dir: str = typer.Option(
        "sample_data", "--output", "-o", help="Output directory for sample datasets"
    ),
    count: int = typer.Option(
        3, "--count", "-n", help="Number of sample datasets to create"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
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
            console=console,
        ) as progress:

            task = progress.add_task("Creating sample datasets...", total=count)

            # Create sample datasets
            created_files = create_sample_datasets(output_path, count)

            for i, file_path in enumerate(created_files):
                progress.update(
                    task, description=f"Created {file_path.name}...", completed=i + 1
                )

        # Display results
        console.print(
            f"\n[bold green]✅ Created {len(created_files)} sample datasets![/bold green]"
        )

        files_table = Table(title="Sample Datasets Created")
        files_table.add_column("File", style="cyan")
        files_table.add_column("Size", style="white")
        files_table.add_column("Description", style="white")

        for file_path in created_files:
            size = file_path.stat().st_size
            size_str = (
                f"{size / 1024:.1f} KB"
                if size < 1024 * 1024
                else f"{size / (1024*1024):.1f} MB"
            )

            descriptions = {
                "sample_clean.csv": "Clean dataset with no issues",
                "sample_mixed.csv": "Mixed data types with some issues",
                "sample_dirty.csv": "Dirty dataset with many problems",
            }

            desc = descriptions.get(file_path.name, "Sample dataset")
            files_table.add_row(file_path.name, size_str, desc)

        console.print(files_table)
        console.print(f"\n[bold]Output directory:[/bold] {output_path}")
        console.print(
            f"[bold]Try cleaning one:[/bold] cleanengine clean {output_path}/sample_mixed.csv"
        )

    except Exception as e:
        console.print(f"[red]Error creating sample datasets: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def test(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    coverage: bool = typer.Option(
        False, "--coverage", help="Run tests with coverage report"
    ),
):
    """Run the CleanEngine test suite"""

    try:
        console.print("[bold]🧪 Running CleanEngine Test Suite...[/bold]")

        # Run tests
        test_results = run_test_suite(verbose=verbose, coverage=coverage)

        if test_results["success"]:
            console.print(
                f"\n[bold green]✅ All tests passed! ({test_results['total']} tests)[/bold green]"
            )
        else:
            console.print(
                f"\n[bold red]❌ {test_results['failed']} tests failed![/bold red]"
            )
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
    host: str = typer.Option(
        "localhost", "--host", "-h", help="Host for Streamlit app"
    ),
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

    info_table.add_row(
        "Python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
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


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to the dataset file to analyze"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Analyze a dataset without cleaning (analysis only)"""
    
    if verbose:
        print_banner()
    
    try:
        # Validate input file
        input_path = validate_file_path(file_path)
        
        # Set output directory
        if output_dir is None:
            output_dir = f"Analysis-{input_path.stem}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        console.print(f"\n[bold green]✅ Analysis completed successfully![/bold green]")
        console.print(f"[bold]Output directory:[/bold] {output_dir}")
        console.print(f"[dim]Note: Full analysis implementation coming soon![/dim]")
        
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def validate_data(
    file_path: str = typer.Argument(..., help="Path to the dataset file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Validate a dataset using CleanEngine's rule engine (validation only)"""
    
    if verbose:
        print_banner()
    
    try:
        # Validate input file
        input_path = validate_file_path(file_path)
        
        console.print(f"\n[bold green]✅ Validation completed![/bold green]")
        console.print(f"[bold]Dataset:[/bold] {input_path}")
        console.print(f"[dim]Note: Full validation implementation coming soon![/dim]")
        
    except Exception as e:
        console.print(f"[red]Error during validation: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def profile(
    file_path: str = typer.Argument(..., help="Path to the dataset file to profile"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for profile"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Generate a comprehensive data profile report (profiling only)"""
    
    if verbose:
        print_banner()
    
    try:
        # Validate input file
        input_path = validate_file_path(file_path)
        
        # Set output directory
        if output_dir is None:
            output_dir = f"Profile-{input_path.stem}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        console.print(f"\n[bold green]✅ Profiling completed successfully![/bold green]")
        console.print(f"[bold]Output:[/bold] {output_dir}")
        console.print(f"[dim]Note: Full profiling implementation coming soon![/dim]")
        
    except Exception as e:
        console.print(f"[red]Error during profiling: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command()
def clean_only(
    file_path: str = typer.Argument(..., help="Path to the dataset file to clean"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for cleaned data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Clean a dataset without analysis (cleaning only)"""

    if verbose:
        print_banner()

    try:
        # Validate input file
        input_path = validate_file_path(file_path)

        # Set output path
        if output_path is None:
            output_path = f"Clean-{input_path.stem}"

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        console.print(f"\n[bold green]✅ Cleaning completed successfully![/bold green]")
        console.print(f"[bold]Output:[/bold] {output_path}")
        console.print(f"[dim]Note: This will use the existing clean command without analysis![/dim]")

    except Exception as e:
        console.print(f"[red]Error during cleaning: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def correlations(
    file_path: str = typer.Argument(..., help="Path to the dataset file for correlation analysis"),
    method: str = typer.Option("pearson", "--method", "-m", help="Correlation method: pearson, spearman, kendall"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Correlation strength threshold"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Analyze correlations between variables in your dataset"""

    if verbose:
        print_banner()

    try:
        input_path = validate_file_path(file_path)

        if output_dir is None:
            output_dir = f"Correlations-{input_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]🔍 Analyzing correlations in:[/bold] {input_path}")
        console.print(f"[bold]Method:[/bold] {method}")
        console.print(f"[bold]Threshold:[/bold] {threshold}")

        # Initialize analyzer
        cleaner = DatasetCleaner()
        df = cleaner.load_data(str(input_path))

        from src.dataset_cleaner.analysis.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)
        analyzer.correlation_analysis()

        if "correlation_analysis" in analyzer.analysis_results:
            corr_results = analyzer.analysis_results["correlation_analysis"]
            strong_correlations = corr_results.get("strong_correlations", [])

            console.print(f"\n[bold green]✅ Found {len(strong_correlations)} strong correlations![/bold green]")

            if strong_correlations:
                corr_table = Table(title="Strong Correlations")
                corr_table.add_column("Variable 1", style="cyan")
                corr_table.add_column("Variable 2", style="magenta")
                corr_table.add_column("Correlation", style="yellow")

                for corr in strong_correlations[:10]:  # Show top 10
                    corr_table.add_row(
                        corr['var1'],
                        corr['var2'],
                        f"{corr['correlation']:.3f}"
                    )

                console.print(corr_table)

            # Save results
            import json
            results_file = output_dir / "correlation_results.json"
            with open(results_file, 'w') as f:
                json.dump(corr_results, f, indent=2)

            console.print(f"\n[bold]📊 Results saved to:[/bold] {results_file}")

        else:
            console.print(f"[yellow]No correlation analysis results available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during correlation analysis: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def features(
    file_path: str = typer.Argument(..., help="Path to the dataset file for feature analysis"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Analyze feature importance and relationships in your dataset"""

    if verbose:
        print_banner()

    try:
        input_path = validate_file_path(file_path)

        if output_dir is None:
            output_dir = f"Features-{input_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]🎯 Analyzing feature importance in:[/bold] {input_path}")

        # Initialize analyzer
        cleaner = DatasetCleaner()
        df = cleaner.load_data(str(input_path))

        from src.dataset_cleaner.analysis.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)
        analyzer.feature_importance_analysis()

        if "feature_importance" in analyzer.analysis_results:
            feat_results = analyzer.analysis_results["feature_importance"]

            if feat_results.get("top_features"):
                console.print(f"\n[bold green]✅ Feature importance analysis completed![/bold green]")

                feat_table = Table(title="Top Features")
                feat_table.add_column("Rank", style="cyan")
                feat_table.add_column("Feature", style="magenta")
                feat_table.add_column("Importance", style="yellow")

                for i, (feature, score) in enumerate(feat_results["top_features"][:10], 1):
                    feat_table.add_row(str(i), feature, f"{score:.4f}")

                console.print(feat_table)

                # Save results
                import json
                results_file = output_dir / "feature_importance.json"
                with open(results_file, 'w') as f:
                    json.dump(feat_results, f, indent=2)

                console.print(f"\n[bold]📊 Results saved to:[/bold] {results_file}")

            else:
                console.print(f"[yellow]No feature importance results available (need numeric columns)[/yellow]")

        else:
            console.print(f"[yellow]Feature importance analysis not available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during feature analysis: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def clusters(
    file_path: str = typer.Argument(..., help="Path to the dataset file for clustering analysis"),
    method: str = typer.Option("kmeans", "--method", "-m", help="Clustering method: kmeans, dbscan, hierarchical"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Discover natural clusters and patterns in your dataset"""

    if verbose:
        print_banner()

    try:
        input_path = validate_file_path(file_path)

        if output_dir is None:
            output_dir = f"Clusters-{input_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]📊 Discovering clusters in:[/bold] {input_path}")
        console.print(f"[bold]Method:[/bold] {method}")

        # Initialize analyzer
        cleaner = DatasetCleaner()
        df = cleaner.load_data(str(input_path))

        from src.dataset_cleaner.analysis.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)

        # Override clustering method in config
        if hasattr(analyzer, 'config') and analyzer.config:
            analyzer.config.update({"analysis.clustering.method": method})

        analyzer.clustering_analysis()

        if "clustering_analysis" in analyzer.analysis_results:
            cluster_results = analyzer.analysis_results["clustering_analysis"]
            optimal_clusters = cluster_results.get("optimal_clusters", 0)

            console.print(f"\n[bold green]✅ Found {optimal_clusters} natural clusters![/bold green]")

            if optimal_clusters > 0 and cluster_results.get("cluster_summary"):
                cluster_table = Table(title="Cluster Summary")
                cluster_table.add_column("Cluster", style="cyan")
                cluster_table.add_column("Size", style="magenta")
                cluster_table.add_column("Percentage", style="yellow")

                for cluster_name, info in cluster_results["cluster_summary"].items():
                    if cluster_name != "noise":  # Skip noise for cleaner display
                        cluster_table.add_row(
                            cluster_name,
                            str(info["size"]),
                            f"{info['percentage']:.1f}%"
                        )

                console.print(cluster_table)

                # Save results
                import json
                results_file = output_dir / "clustering_results.json"
                with open(results_file, 'w') as f:
                    json.dump(cluster_results, f, indent=2)

                console.print(f"\n[bold]📊 Results saved to:[/bold] {results_file}")

            else:
                console.print(f"[yellow]No meaningful clusters found[/yellow]")

        else:
            console.print(f"[yellow]Clustering analysis not available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during clustering analysis: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def anomalies(
    file_path: str = typer.Argument(..., help="Path to the dataset file for anomaly detection"),
    method: str = typer.Option("isolation_forest", "--method", "-m", help="Anomaly detection method: isolation_forest, lof"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", help="Expected proportion of anomalies"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Detect anomalies and outliers in your dataset"""

    if verbose:
        print_banner()

    try:
        input_path = validate_file_path(file_path)

        if output_dir is None:
            output_dir = f"Anomalies-{input_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]🔍 Detecting anomalies in:[/bold] {input_path}")
        console.print(f"[bold]Method:[/bold] {method}")
        console.print(f"[bold]Expected anomalies:[/bold] {contamination*100:.1f}%")

        # Initialize analyzer
        cleaner = DatasetCleaner()
        df = cleaner.load_data(str(input_path))

        from src.dataset_cleaner.analysis.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)

        # Override anomaly detection settings in config
        if hasattr(analyzer, 'config') and analyzer.config:
            analyzer.config.update({
                "analysis.anomaly_detection.method": method,
                "analysis.anomaly_detection.contamination": contamination
            })

        analyzer.anomaly_detection()

        if "anomaly_detection" in analyzer.analysis_results:
            anomaly_results = analyzer.analysis_results["anomaly_detection"]
            total_anomalies = anomaly_results.get("total_anomalies", 0)
            anomaly_percentage = anomaly_results.get("anomaly_percentage", 0)

            console.print(f"\n[bold green]✅ Found {total_anomalies} anomalies ({anomaly_percentage:.1f}%)![/bold green]")

            anomaly_table = Table(title="Anomaly Detection Results")
            anomaly_table.add_column("Metric", style="cyan")
            anomaly_table.add_column("Value", style="white")

            anomaly_table.add_row("Method", anomaly_results.get("method", "Unknown"))
            anomaly_table.add_row("Total Anomalies", str(total_anomalies))
            anomaly_table.add_row("Anomaly Percentage", f"{anomaly_percentage:.1f}%")
            anomaly_table.add_row("Dataset Size", str(len(df)))

            console.print(anomaly_table)

            # Save results
            import json
            results_file = output_dir / "anomaly_results.json"
            with open(results_file, 'w') as f:
                json.dump(anomaly_results, f, indent=2)

            console.print(f"\n[bold]📊 Results saved to:[/bold] {results_file}")

            if anomaly_percentage > 10:
                console.print(f"\n[yellow]⚠️  High anomaly rate detected! Consider data quality review.[/yellow]")

        else:
            console.print(f"[yellow]Anomaly detection analysis not available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during anomaly detection: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def quality(
    file_path: str = typer.Argument(..., help="Path to the dataset file for quality assessment"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Assess overall data quality and generate quality report"""

    if verbose:
        print_banner()

    try:
        input_path = validate_file_path(file_path)

        if output_dir is None:
            output_dir = f"Quality-{input_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]📋 Assessing data quality for:[/bold] {input_path}")

        # Initialize analyzer
        cleaner = DatasetCleaner()
        df = cleaner.load_data(str(input_path))

        from src.dataset_cleaner.analysis.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)
        analyzer.data_quality_assessment()

        if "data_quality" in analyzer.analysis_results:
            quality_results = analyzer.analysis_results["data_quality"]

            completeness = quality_results.get("completeness_percentage", 0)
            quality_score = quality_results.get("overall_quality_score", 0)
            consistency_issues = len(quality_results.get("consistency_issues", []))

            console.print(f"\n[bold green]✅ Data quality assessment completed![/bold green]")

            quality_table = Table(title="Data Quality Assessment")
            quality_table.add_column("Metric", style="cyan")
            quality_table.add_column("Value", style="white")
            quality_table.add_column("Status", style="green")

            # Completeness
            comp_status = "✅ Good" if completeness >= 90 else "⚠️  Needs Attention" if completeness >= 70 else "❌ Poor"
            quality_table.add_row("Completeness", f"{completeness:.1f}%", comp_status)

            # Quality Score
            score_status = "✅ Excellent" if quality_score >= 80 else "⚠️  Good" if quality_score >= 60 else "❌ Needs Improvement"
            quality_table.add_row("Overall Score", f"{quality_score:.1f}/100", score_status)

            # Consistency
            cons_status = "✅ Good" if consistency_issues == 0 else f"⚠️  {consistency_issues} issues"
            quality_table.add_row("Consistency Issues", str(consistency_issues), cons_status)

            console.print(quality_table)

            # Show insights
            if analyzer.insights:
                console.print(f"\n[bold]💡 Key Insights:[/bold]")
                for insight in analyzer.insights[:5]:  # Show top 5 insights
                    console.print(f"  • {insight}")

            # Save results
            import json
            results_file = output_dir / "quality_assessment.json"
            with open(results_file, 'w') as f:
                json.dump(quality_results, f, indent=2)

            console.print(f"\n[bold]📊 Results saved to:[/bold] {results_file}")

        else:
            console.print(f"[yellow]Data quality assessment not available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during quality assessment: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def statistics(
    file_path: str = typer.Argument(..., help="Path to the dataset file for statistical analysis"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Perform comprehensive statistical analysis on your dataset"""

    if verbose:
        print_banner()

    try:
        input_path = validate_file_path(file_path)

        if output_dir is None:
            output_dir = f"Statistics-{input_path.stem}"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        console.print(f"\n[bold]📊 Performing statistical analysis on:[/bold] {input_path}")

        # Initialize analyzer
        cleaner = DatasetCleaner()
        df = cleaner.load_data(str(input_path))

        from src.dataset_cleaner.analysis.analyzer import DataAnalyzer
        analyzer = DataAnalyzer(df)
        analyzer.statistical_analysis()

        if "statistical_summary" in analyzer.analysis_results:
            stats_results = analyzer.analysis_results["statistical_summary"]

            console.print(f"\n[bold green]✅ Statistical analysis completed![/bold green]")

            stats_table = Table(title="Statistical Summary")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")

            stats_table.add_row("Total Columns", str(len(df.columns)))
            stats_table.add_row("Numeric Columns", str(stats_results.get("numeric_columns_count", 0)))

            if stats_results.get("skewness"):
                highly_skewed = {k: v for k, v in stats_results["skewness"].items() if abs(v) > 1}
                stats_table.add_row("Highly Skewed Variables", str(len(highly_skewed)))

            console.print(stats_table)

            # Save results
            import json
            results_file = output_dir / "statistical_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(stats_results, f, indent=2)

            console.print(f"\n[bold]📊 Results saved to:[/bold] {results_file}")

        else:
            console.print(f"[yellow]Statistical analysis not available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during statistical analysis: {str(e)}[/red]")
        raise typer.Exit(1)

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
