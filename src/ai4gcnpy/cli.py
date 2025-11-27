from . import llm_client
from .agents import CircularState, GCNExtractorAgent

from pathlib import Path
from typing import Optional, Literal
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
import logging
import logging.config
from dotenv import load_dotenv

load_dotenv()

# Global console and logger
logger = logging.getLogger(__name__)
console = Console()


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
def Extractor(
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
    # Read input file
    try:
        text = Path(input_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.error("Cannot read input file %s: %s", input_file, e)
        raise typer.Exit(code=1)

    llm_config = {
        "model": model,
        "model_provider": model_provider,
    }
    if temperature is not None:
        llm_config["temperature"] = temperature
    if max_tokens is not None:
        llm_config["max_tokens"] = max_tokens
    if reasoning is not None:
        llm_config["reasoning"] = reasoning
    llm_client.basicConfig(**llm_config)

    # Compile into a runnable app
    app = GCNExtractorAgent()

    # # Run the workflow
    initial_state = CircularState(raw_text=text)
    final_state = app.invoke(initial_state)

    # # Output results
    console.print(Rule("Circular", style="dim"))
    console.print(final_state["raw_text"])
    console.print(Rule("Extracted Results", style="dim"))
    console.print(final_state["extracted_dset"])