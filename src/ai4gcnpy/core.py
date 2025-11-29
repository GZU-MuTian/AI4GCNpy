from . import llm_client
from .agents import CircularState, GCNExtractorAgent

from typing import List, Dict, Any, Literal, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


def _run_extraction(
    input_file: str,
    model: str = "deepseek-chat",
    model_provider: str = "deepseek",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    # Read input file
    try:
        text = Path(input_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.error("Cannot read input file %s: %s", input_file, e)
        return {}

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
    final_state: dict = app.invoke(initial_state)

    return final_state