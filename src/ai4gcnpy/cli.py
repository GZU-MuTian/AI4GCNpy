from .core import _run_extraction

from typing import Optional, Literal
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.json import JSON
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

