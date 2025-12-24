from .core import _run_extraction, _run_builder, _run_graphrag
from .utils import download_gcn_archive

from typing import Optional, Literal, List
import typer
import json
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.json import JSON
from rich.rule import Rule
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import track
from pathlib import Path
import logging
import logging.config

# Global console and logger
logger = logging.getLogger(__name__)
console = Console(highlight=False)


# --- CLI Command ---

app = typer.Typer(
    help="AI for NASA GCN Circulars",
    rich_markup_mode="rich"
)

# Define allowed log levels using Literal
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

@app.callback()
def main(
    log_level: LogLevel = typer.Option("ERROR", "--log-level", "-v", help="Enable logging with specified level (e.g., DEBUG, INFO, WARNING). ERROR by default."),
) -> None:
    """
    Global options for all commands.
    
    The --log-level option applies to every subcommand automatically.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            }
        },
        "handlers": {
            "rich": {
                "()": RichHandler,
                "level": log_level,
                "rich_tracebacks": False,
                "show_time": True,
                "show_path": True,
                "formatter": "default",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["rich"],
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)

@app.command(help="Extract structured information from a GCN Circular using an LLM.")
def extractor(
    input_file: str = typer.Argument(..., help="Path to a text file containing GCN Circular content."),
    model: str = typer.Option("deepseek-chat", "--model", "-m", help="Model name."),
    model_provider: str = typer.Option("deepseek", "--provider", "-p", help="Model provider."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Sampling temperature."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum number of output tokens."),
    reasoning: Optional[bool] = typer.Option(None, "--reasoning", help="Controls the reasoning/thinking mode for supported models."),
) -> None:
    """
    Main CLI entry point to run the GCN extractor.
    """
    results = _run_extraction(
        input_file=input_file,
        model=model,
        model_provider=model_provider,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    # Output results
    console.print(Panel(
        results.get("raw_text", ""),
        title="Circular", 
    ))
    extracted_dset = results.get("extracted_dset", {})
    console.print(Panel(
        JSON.from_data(extracted_dset),
        title="Extraction Result", 
    ))

@app.command(help="Batch extract from one or more GCN Circular TXT files. If no input is given, downloads data automatically.")
def batch_extractor(
        input_path: Optional[str] = typer.Argument(
            None,
            help="Path to a TXT file or a directory containing TXT files. If omitted, data will be downloaded and processed."
        ),
        output_path: Optional[str] = typer.Argument(
            None,
            help="Directory to save extraction results as JSON files. Defaults to GCN_OUTPUT_PATH env var or './output'."
        ),
        model: str = typer.Option("deepseek-chat", "--model", "-m", help="Model name."),
        model_provider: str = typer.Option("deepseek", "--provider", "-p", help="Model provider."),
        temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Sampling temperature."),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum number of output tokens."),
        reasoning: Optional[bool] = typer.Option(None, "--reasoning", help="Enable reasoning mode if supported."),
    ) -> None:
    """
    Enhanced extractor that supports batch processing and auto-download.
    """
    GCN_URL = "https://gcn.nasa.gov/circulars/archive.txt.tar.gz"
    # Step 1: Resolve input source
    if input_path is None:
        logger.info(f"No input path provided. Downloading dataset from {GCN_URL}.")
        input_path = download_gcn_archive(GCN_URL)

    try:
        path_obj = Path(input_path).resolve()
        if path_obj.is_file():
            if path_obj.suffix.lower() == '.txt':
                txt_files: List[Path] = [path_obj]
            else:
                logger.error(f"Input file is not a .txt file: {input_path}")     
        elif path_obj.is_dir():
            txt_files = sorted(path_obj.rglob("*.txt"))
            logger.debug(f"Found {len(txt_files)} TXT file(s) in: {input_path}")
        else:
            logger.error(f"Path is neither a file nor a directory: {input_path}")      
    except Exception as e:
        logger.exception(f"Error processing input path: {input_path}: {e}")

    # --- Step 2: Resolve output path ---
    if output_path is None:
        output_path = os.getenv("GCN_OUTPUT_PATH", "./output")
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Step 3: Process each file, skip if already exists ---
    files_processed = 0
    for txt_file in track(txt_files, description="Processing files...", transient=True):
        stem = txt_file.stem
        json_file = output_dir / f"{stem}.json"
        if json_file.exists():
            continue
    
        try:
            result = _run_extraction(
                str(txt_file),
                model=model,
                model_provider=model_provider,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            # Write result as JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            files_processed += 1
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {str(e)}")
            continue

    console.print(f"Files Processed: {files_processed}")

@app.command(help="Build a GCN knowledge graph from structured extraction results.")
def builder(
    input_path: str = typer.Argument(..., help="Path to a JSON file or a directory containing JSON files."),
    database: str = typer.Option("neo4j", "--database", "-d", help="Neo4j database name."),
) -> None:
    """
    Main CLI entry point for building a GCN graph database.
    """
    try:
        path_obj = Path(input_path).resolve()
        # Resolve to list of JSON files
        if path_obj.is_file():
            if path_obj.suffix.lower() == '.json':
                json_files: List[Path] = [path_obj]
            else:
                logger.error(f"Input file is not a JSON file: {input_path}")
        elif path_obj.is_dir():
            json_files = sorted(path_obj.rglob("*.json"))
            logger.debug(f"Found {len(json_files)} JSON file(s) in: {input_path}")
            if not json_files:
                logger.warning(f"No JSON files found in directory: {input_path}")
        else:
            logger.error(f"Path is neither a file nor a directory: {input_path}")
    except Exception as e:
        logger.exception(f"Error processing input path: {input_path}: {e}")

    files_processed = 0
    for json_file in track(json_files, description="Processing files...", transient=True):
        if _run_builder(json_file=json_file.as_posix(), database=database):
            files_processed += 1
        else:
            logger.warning(f"Skipped or failed processing: {json_file}")

    # Display results using Rich
    console.print(f"Files Processed: {files_processed}/{len(json_files)}")


@app.command(help="Query the GraphRAG knowledge graph for natural language answers.")
def query(
    query_text: str = typer.Argument(..., help="The user's query text to process."),
    model: str = typer.Option("deepseek-chat", "--model", "-m", help="Model name."),
    model_provider: str = typer.Option("deepseek", "--provider", "-p", help="Model provider."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Sampling temperature."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum number of output tokens."),
    reasoning: Optional[bool] = typer.Option(None, "--reasoning", help="Controls the reasoning/thinking mode for supported models."),
    url: Optional[str] = typer.Option(None, "--url", help="Neo4j database URL (e.g., bolt://localhost:7687)."),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="Neo4j username."),
    password: Optional[str] = typer.Option(None, "--password", help="Neo4j password."),
    database: str = typer.Option("neo4j", "--database", "-d", help="Target database name (default: neo4j).")
) -> None:
    """
    Main CLI entry point for execute GraphRAG queries against the knowledge graph.
    """
    try:
        response = _run_graphrag(
            query_text, 
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            url=url,
            username=username,
            password=password,
            database=database
        )

        # --- 1. User Query ---
        console.print("[bold blue]User Query:[/bold blue]")
        console.print(response.get("query"), style="italic")

        # --- 2. Generated Cypher ---
        cypher = response.get("cypher_statement")
        if cypher:
            console.print(Syntax(cypher, "cypher", theme="monokai", word_wrap=True))

        # --- 3. Final Answer (Markdown) ---
        answer = response.get("answer")
        if answer:
            console.print(Rule("[bold]Final Answer"))
            console.print(Markdown(answer))

        # --- 4. Evidence / Retrieved Chunks (Data Sources) ---
        retrieved_chunks = response.get("retrieved_chunks")
        if retrieved_chunks is not None:
            evidence_md_lines = []
            for i, rec in enumerate(retrieved_chunks, 1):
                rec_str = "\n".join(f"  - **{k}**: `{v}`" for k, v in rec.items())
                evidence_md_lines.append(f"> **Record {i}:**\n{rec_str}")
            evidence_md = "\n\n".join(evidence_md_lines)
            console.print(Markdown(evidence_md))

    except Exception as e:
        logger.error("Query command failed: %s", str(e))
        return None

