from . import llm_client
from .agents import CircularState, GCNExtractorAgent

from typing import Dict, Any, Optional
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
    """
    Execute the GCN extraction workflow on an input file.

    Args:
        input_file: Path to the input text file (UTF-8 encoded).
        model: LLM model name to use for extraction.
        model_provider: Provider of the LLM service.
        temperature: Sampling temperature for LLM (0.0-2.0).
        max_tokens: Maximum tokens in LLM response.
        reasoning: Whether to enable chain-of-thought reasoning.

    Returns:
        Extracted structured data as dictionary, or empty dict on failure.
    """
    # Read input file
    try:
        text = Path(input_file).read_text(encoding="utf-8")
    except Exception as e:
        logger.error("pathlib.Path | %s", e)
        return {}

    llm_config: Dict[str, Any] = {
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
    
    try:
        # Compile into a runnable app
        app = GCNExtractorAgent()

        # # Run the workflow
        initial_state = CircularState(raw_text=text)
        final_state: dict = app.invoke(initial_state)
    except Exception as e:
        logger.error(f"GCNExtractorAgent execution failed: {e}")
        return {}

    return final_state