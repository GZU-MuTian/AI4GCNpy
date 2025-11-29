from .core import _run_extraction

from typing import Optional, Literal
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
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
    log_level: Optional[LogLevel] = typer.Option(None, "--log-level", "-v", help="Enable logging with specified level (e.g., DEBUG, INFO, WARNING). Disabled by default."),
) -> None:
    """
    Global options for all commands.
    
    The --log-level option applies to every subcommand automatically.
    """
    if log_level is None:
        logging.disable(logging.WARNING)
        return

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
                "()": RichHandler,               # ← 关键：指定 handler 类
                "level": log_level,
                "rich_tracebacks": True,
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
    try:
        results = _run_extraction(
            input_file=input_file,
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        raise typer.Exit(code=1)

    # Output results
    console.print(Rule("Circular", style="dim"))
    console.print(results["raw_text"])
    console.print(Rule("Extraction Result", style="dim"))
    console.print(JSON.from_data(results["extracted_dset"]))