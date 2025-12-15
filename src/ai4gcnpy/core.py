from . import llm_client
from .agents import CircularState, GCNExtractorAgent
from .db_client import GCNGraphDB
from .utils import build_cypher_statements

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
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


def _run_builder(
    input_path: str,
    database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a GCN knowledge graph in the specified database from one or more extraction result files.

    Args:
        input_path: Path to a JSON file or a directory containing JSON files.
        database: Database identifier.
    """
    # Initialize consistent return structure
    count = {
        "files_processed": 0,
        "files_skipped": 0,
    }

    try:
        path_obj = Path(input_path).resolve()
        
        # Resolve to list of JSON files
        if path_obj.is_file():
            if path_obj.suffix.lower() == '.json':
                json_files: List[Path] = [path_obj]
            else:
                logger.error(f"Input file is not a JSON file: {input_path}")
                return count
        elif path_obj.is_dir():
            json_files = sorted(path_obj.rglob("*.json"))
            logger.debug(f"Found {len(json_files)} JSON file(s) in: {input_path}")
            if not json_files:
                logger.warning(f"No JSON files found in directory: {input_path}")
        else:
            logger.error(f"Path is neither a file nor a directory: {input_path}")
            return count
    except Exception as e:
        logger.exception(f"Error processing input path: {input_path}")
        return count
    
    # Ingest all files inside a database transaction
    graoh = GCNGraphDB()
    with graoh.transaction(database) as tx:
        for json_file in json_files:
            logger.debug(f"Processing file: {json_file}")
            try:
                raw_text = json_file.read_text(encoding="utf-8")
                payload = json.loads(raw_text)
            except Exception as e:
                count["files_skipped"] += 1
                logger.error(f"Failed to read or parse JSON file {json_file}: {e}")
                continue  # Skip malformed files

            if not payload:
                count["files_skipped"] += 1
                logger.debug(f"Empty payload in file: {json_file}, skipping.")
                continue

            try:
                cypher_statements = build_cypher_statements(payload)
                for query, params in cypher_statements:
                    tx.run(query, params)
                count["files_processed"] += 1
            except Exception as e:
                count["files_skipped"] += 1
                logger.error(f"Failed to generate/run Cypher for {json_file}: {e}")
                continue

        logger.debug("All files processed successfully. Transaction committed.")
    graoh.close()

    return count
