from .core import _run_extraction, _run_builder, _run_graphrag

from typing import Optional, Literal, List
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.json import JSON
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
    except Exception as e:
        logger.error("Query command failed: %s", str(e))
        return None

    console.print(Panel(
        response["cypher_statement"],
        title="Cypher", 
    ))

    console.print(Panel(
        response["retrieved_chunks"],
        title="Cypher", 
    ))