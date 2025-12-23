from . import llm_client
from .agents import CircularState, GraphQAState, GCNExtractorAgent, GraphQAAgent
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
    json_file: str,
    database: Optional[str] = None,
) -> bool:
    """
    Build a GCN knowledge graph in the specified database from one or more extraction result files.

    Args:
        json_file: Path to a JSON file.
        database: Database identifier.
    """
    logger.debug(f"Processing file: {json_file}")
    try:
        raw_text = Path(json_file).read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file {json_file}: {e}")
        return False
    if not payload:
        logger.debug(f"Empty payload in file: {json_file}, skipping.")
        return False

    graph = GCNGraphDB()
    with graph.transaction(database) as tx:
        try:
            cypher_statements = build_cypher_statements(payload)
            for query, params in cypher_statements:
                tx.run(query, params)
        except Exception as e:
            logger.error(f"Failed to generate/run Cypher for {json_file}: {e}")
            return False
    graph.close()
    return True

def _run_graphrag(
    query_text: str,
    model: str = "deepseek-chat",
    model_provider: str = "deepseek",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[bool] = None,
    url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,    
):
    """
    Process a user query against the knowledge graph database.
    
    Args:
        query_text: The natural language query from the user.
        database: The target database name (default: 'neo4j').
        
    Returns:
        str: Contains the generated answer and optional context.
    """
    # Input validation
    if not query_text.strip():
        raise ValueError("Query text cannot be empty or whitespace-only")

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
        graph = GCNGraphDB(url=url, username=username, password=password)

        # Compile into a runnable app
        app = GraphQAAgent()
        # Run the workflow
        initial_state = GraphQAState(
            query=query_text, 
            graph=graph, 
            database=database
        )

        final_state = app.invoke(initial_state)

        graph.close()

        return final_state
    except Exception as e:
        logger.error(f"GCNExtractorAgent execution failed: {e}")
        return {}

    
    
