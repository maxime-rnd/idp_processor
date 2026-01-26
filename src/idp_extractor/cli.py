"""Command Line Interface for IDP Extractor."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import extract_passport_info
from .config import Settings

app = typer.Typer(
    name="idp-extractor",
    help="Intelligent Document Processing extractor for French passports",
    add_completion=False,
)
console = Console()


def get_settings() -> Settings:
    """Get settings from environment and .env file."""
    return Settings()


@app.command()
def process(
    file_path: Path = typer.Argument(..., help="Path to image or PDF file to process"),
    api_key: Optional[str] = typer.Option(None, help="API key (overrides .env setting)"),
    output: Optional[Path] = typer.Option(None, help="Output JSON file path"),
    verbose: bool = typer.Option(False, help="Enable verbose output"),
) -> None:
    """Process a single file and extract passport information."""
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File '{file_path}' does not exist.")
        raise typer.Exit(1)

    settings = get_settings()
    final_api_key = api_key or settings.api_key

    if not final_api_key:
        console.print("[red]Error:[/red] No API key provided. Set API_KEY in .env file or use --api-key option.")
        console.print("[dim]Example: cp .env.example .env && edit .env[/dim]")
        raise typer.Exit(1)

    if verbose:
        console.print(f"Processing file: {file_path}")
        console.print(f"Using model: {settings.model}")

    try:
        with console.status(f"Extracting passport information from {file_path.name}..."):
            result = extract_passport_info(str(file_path), api_key=final_api_key)

        if result.success and result.passport:
            console.print("[green]✓[/green] Successfully extracted passport information")

            if verbose:
                # Display results in a nice table
                table = Table(title="Extracted Passport Information")
                table.add_column("Field", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")

                passport_data = result.passport.model_dump()
                for field, value in passport_data.items():
                    if field == "mrz" and value:
                        table.add_row("MRZ Line 1", value.get("line_1", "N/A"))
                        table.add_row("MRZ Line 2", value.get("line_2", "N/A"))
                    elif field != "mrz":
                        table.add_row(field.replace("_", " ").title(), str(value))

                console.print(table)

                # Display metrics
                if result.metrics:
                    metrics_table = Table(title="Performance Metrics")
                    metrics_table.add_column("Metric", style="cyan")
                    metrics_table.add_column("Value", style="yellow")

                    # Get metrics data including computed properties
                    metrics_dict = result.metrics.model_dump()
                    metrics_dict['cost_eur'] = result.metrics.cost_eur
                    metrics_dict['confidence'] = result.metrics.confidence
                    
                    for field, value in metrics_dict.items():
                        if field == "cost_eur":
                            metrics_table.add_row("Cost (EUR)", ".6f")
                        elif field == "confidence" and value is not None:
                            metrics_table.add_row("Confidence", ".1%")
                        elif field not in ["timestamp", "_settings"]:  # Skip internal fields
                            metrics_table.add_row(field.replace("_", " ").title(), str(value))

                    console.print(metrics_table)
        else:
            console.print(f"[red]✗[/red] Failed to extract passport information: {result.error_message}")

        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[green]Results saved to:[/green] {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def process(
    file_path: Path = typer.Argument(..., help="Path to image or PDF file to process"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key (can also be set via OPENAI_API_KEY env var)"),
    output: Optional[Path] = typer.Option(None, help="Output JSON file path"),
    verbose: bool = typer.Option(False, help="Enable verbose output"),
) -> None:
    """Process a single file and extract passport information."""
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File '{file_path}' does not exist.")
        raise typer.Exit(1)

    if verbose:
        console.print(f"Processing file: {file_path}")

    try:
        with console.status(f"Extracting passport information from {file_path.name}..."):
            result = extract_passport_info(str(file_path), api_key=api_key)

        if result.success and result.passport:
            console.print("[green]✓[/green] Successfully extracted passport information")

            if verbose:
                # Display results in a nice table
                table = Table(title="Extracted Passport Information")
                table.add_column("Field", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")

                passport_data = result.passport.model_dump()
                for field, value in passport_data.items():
                    if field == "mrz" and value:
                        table.add_row("MRZ Line 1", value.get("line_1", "N/A"))
                        table.add_row("MRZ Line 2", value.get("line_2", "N/A"))
                    elif field != "mrz":
                        table.add_row(field.replace("_", " ").title(), str(value))

                console.print(table)

                # Display metrics
                if result.metrics:
                    metrics_table = Table(title="Performance Metrics")
                    metrics_table.add_column("Metric", style="cyan")
                    metrics_table.add_column("Value", style="yellow")

                    metrics_data = result.metrics.model_dump()
                    for field, value in metrics_data.items():
                        if field == "cost_eur":
                            metrics_table.add_row("Cost (EUR)", ".6f")
                        elif field == "confidence" and value is not None:
                            metrics_table.add_row("Confidence", ".1%")
                        else:
                            metrics_table.add_row(field.replace("_", " ").title(), str(value))

                    console.print(metrics_table)
        else:
            console.print(f"[red]✗[/red] Failed to extract passport information: {result.error_message}")

        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[green]Results saved to:[/green] {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def process_folder(
    folder_path: Path = typer.Argument(..., help="Path to folder containing images/PDFs"),
    output: Path = typer.Option("results.json", help="Output JSON file path"),
    api_key: Optional[str] = typer.Option(None, help="API key (overrides .env setting)"),
    recursive: bool = typer.Option(True, help="Process subfolders recursively"),
    verbose: bool = typer.Option(False, help="Enable verbose output"),
) -> None:
    """Process all images and PDFs in a folder."""
    if not folder_path.exists():
        console.print(f"[red]Error:[/red] Folder '{folder_path}' does not exist.")
        raise typer.Exit(1)

    if not folder_path.is_dir():
        console.print(f"[red]Error:[/red] '{folder_path}' is not a directory.")
        raise typer.Exit(1)

    settings = get_settings()
    final_api_key = api_key or settings.api_key

    if not final_api_key:
        console.print("[red]Error:[/red] No API key provided. Set API_KEY in .env file or use --api-key option.")
        raise typer.Exit(1)

    if verbose:
        console.print(f"Using model: {settings.model}")
        console.print(f"Cost per token: {settings.cost_per_token} EUR")

    # Find all files to process
    if recursive:
        files_to_process = [
            f for f in folder_path.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]
    else:
        files_to_process = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

    if not files_to_process:
        console.print(f"[yellow]Warning:[/yellow] No supported files found in {folder_path}")
        return

    console.print(f"Found {len(files_to_process)} files to process")

    results = []
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=len(files_to_process))

        for file_path in files_to_process:
            try:
                progress.update(task, description=f"Processing {file_path.name}...")
                result = extract_passport_info(str(file_path), api_key=final_api_key)

                result_dict = {
                    "file": str(file_path),
                    "success": result.success,
                    "passport": result.passport.model_dump() if result.passport else None,
                    "error": result.error_message,
                    "model_used": result.model_used,
                    "metrics": result.metrics.model_dump() if result.metrics else None,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                }
                results.append(result_dict)

                if result.success:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                console.print(f"[red]Error processing {file_path}:[/red] {e}")
                results.append({
                    "file": str(file_path),
                    "success": False,
                    "error": str(e),
                })
                failed += 1

            progress.update(task, advance=1)

    # Save results
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    console.print(f"\n[green]✓ Processing complete![/green]")
    console.print(f"Results saved to: {output}")
    console.print(f"Successful: {successful}")
    console.print(f"Failed: {failed}")
    console.print(f"Total: {len(results)}")

    if verbose and results:
        # Show summary table
        summary_table = Table(title="Processing Summary")
        summary_table.add_column("File", style="cyan", no_wrap=True)
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Cost (EUR)", style="yellow", justify="right")

        for result in results[:10]:  # Show first 10 results
            status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
            cost = ".6f" if result.get("metrics", {}).get("cost_eur") else "N/A"
            summary_table.add_row(
                Path(result["file"]).name,
                status,
                cost
            )

        if len(results) > 10:
            summary_table.add_row("...", "...", "...")

        console.print(summary_table)


@app.callback()
def main() -> None:
    """IDP Extractor - Extract information from French passports using AI."""
    pass


if __name__ == "__main__":
    app()