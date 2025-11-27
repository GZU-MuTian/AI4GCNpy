from .chains import TopicLabelerChain
from .utils import split_text_into_paragraphs, group_paragraphs_by_labels, header_regex_match

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import partial

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)


# --- State Definition ---

class CircularState(BaseModel):
    raw_text: str = Field(..., description="Original GCN circular text.")
    paragraphs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Paragraphs assigned topic label."
    )
    pending_labels: List[str] = Field(default_factory=list, description="Keys left to process.")
    extracted_dset: Dict[str, Any] = Field(default_factory=dict, description="Dict storing extracted circular information.")

# --- Node Functions ---

def text_split(state: CircularState) -> Dict[str, Any]:
    """
    Assign topic labels to paragraphs using an LLM.
    
    Args:
        state (CircularState): The current state containing raw_text.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'paragraphs' key.
    """
    raw_text = state.raw_text
    
    # Split
    paragraphs = split_text_into_paragraphs(raw_text)
    if not paragraphs:
        logger.warning("No paragraphs found in input text.")
        return {}

    # Prepare input data with clear prefix P<N>
    numbered_paragraphs_parts = [f"<P{i+1}>{p}</P{i+1}>" for i, p in enumerate(paragraphs)]
    numbered_paragraphs_str = "\n".join(numbered_paragraphs_parts)
    logger.debug(numbered_paragraphs_str)

    # Label using LLM
    logger.debug("Labeling paragraphs with topics.")
    chain = TopicLabelerChain()

    try:
        responses = chain.invoke({"numbered_paragraphs": numbered_paragraphs_str})
        logger.debug("TopicLabelerChain completed")
    except Exception as e:
        logger.error(f"Failed to label topic: {e}", exc_info=True)
        return {}

    labeled_paragraphs = group_paragraphs_by_labels(paragraphs, responses.labels)

    return {"paragraphs": labeled_paragraphs, "pending_labels": labeled_paragraphs.keys()}

def router_node(state: CircularState)  -> str:
    """
    Routing function that determines the next node to execute based on the current 'pending_labels' list.
    
    - If the 'pending_labels' list is empty, the workflow should end, so return 'end'.
    - Otherwise, return the first task in the 'pending_labels' list, which must match the name of a registered node.
    """
    if not state["pending_labels"]:
        logger.debug("Router: No pending labels — exiting loop.")
        return "end_loop"
    next = state["pending_labels"][0]
    logger.debug(f"Router: Selected next node '{next}'")
    return next

# --- Extractor Nodes ---

def extract_header_information(state: CircularState) -> Dict[str, Any]:
    """
    Extracts GCN Circular header information from the 'HeaderInformation' paragraph.

    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """
    paragraph = state.paragraphs.get("HeaderInformation", "")
    if not paragraph.strip():
        raise ValueError("HeaderInformation paragraph is empty or missing.")

    # parse GCN Circular header
    extracted_info = header_regex_match(paragraph)
    logger.debug("Successfully extracted information: %s", extracted_info.model_dump())

    # Update extracted dataset
    current_extracted = state.get("extracted_dset", {})
    updated_extracted = {**current_extracted, **extracted_info.model_dump()}

    # Remove the processed label
    updated_pending = state["pending_labels"][1:]
    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }

def extract_author_list(state: CircularState) -> Dict[str, Any]:
    """
    Simulates extraction of author list information.

    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """
    paragraph = state.paragraphs.get("AuthorList", "")
    if not paragraph.strip():
        raise ValueError("AuthorList paragraph is empty or missing.")

    # parse GCN Circular author list
    extracted_info = {"author_list_sample": "example"}
    logger.debug("Successfully extracted information: %s", extracted_info.model_dump())

    # Update extracted dataset
    current_extracted = state.get("extracted_dset", {})
    updated_extracted = {**current_extracted, **extracted_info.model_dump()}

    # Remove the processed label
    updated_pending = state["pending_labels"][1:]
    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }


def extract_scientific_content(state: CircularState) -> Dict[str, Any]:
    """
    Simulates extraction of scientific content details.

    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """ 
    paragraph = state.paragraphs.get("ScientificContent", "")
    if not paragraph.strip():
        raise ValueError("ScientificContent paragraph is empty or missing.")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"scientific_content_sample": "example"}
    logger.debug("Successfully extracted information: %s", extracted_info.model_dump())

    # Update extracted dataset
    current_extracted = state.get("extracted_dset", {})
    updated_extracted = {**current_extracted, **extracted_info.model_dump()}

    # Remove the processed label
    updated_pending = state["pending_labels"][1:]
    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }


def extract_references(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """ 
    paragraph = state.paragraphs.get("References", "")
    if not paragraph.strip():
        raise ValueError("References paragraph is empty or missing.")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"references_sample": "example"}
    logger.debug("Successfully extracted information: %s", extracted_info.model_dump())

    # Update extracted dataset
    current_extracted = state.get("extracted_dset", {})
    updated_extracted = {**current_extracted, **extracted_info.model_dump()}

    # Remove the processed label
    updated_pending = state["pending_labels"][1:]
    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }


def extract_contact_information(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """ 
    paragraph = state.paragraphs.get("ContactInformation", "")
    if not paragraph.strip():
        raise ValueError("ContactInformation paragraph is empty or missing.")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"contact_information_sample": "example"}
    logger.debug("Successfully extracted information: %s", extracted_info.model_dump())

    # Update extracted dataset
    current_extracted = state.get("extracted_dset", {})
    updated_extracted = {**current_extracted, **extracted_info.model_dump()}

    # Remove the processed label
    updated_pending = state["pending_labels"][1:]
    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }


def extract_acknowledgements(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """ 
    paragraph = state.paragraphs.get("Acknowledgements", "")
    if not paragraph.strip():
        raise ValueError("Acknowledgements paragraph is empty or missing.")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"acknowledgements_sample": "example"}
    logger.debug("Successfully extracted information: %s", extracted_info.model_dump())

    # Update extracted dataset
    current_extracted = state.get("extracted_dset", {})
    updated_extracted = {**current_extracted, **extracted_info.model_dump()}

    # Remove the processed label
    updated_pending = state["pending_labels"][1:]
    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }


# --- Graph Construction ---

def GCNExtractorAgent():
    """
    Agent that processes a GCN Circular text and returns structured data.

    Returns:
        StateGraph: The compiled workflow graph.
    """
    logging.debug("Creating LangGraph workflow.")
    # Initialize the state graph 
    workflow = StateGraph(CircularState)

    # Add nodes
    workflow.add_node("text_split", text_split)
    workflow.add_node("router_node", router_node)
    workflow.add_node("extract_header_information", extract_header_information)
    workflow.add_node("extract_author_list", extract_author_list)
    workflow.add_node("extract_scientific_content", extract_scientific_content)
    workflow.add_node("extract_references", extract_references)
    workflow.add_node("extract_contact_information", extract_contact_information)
    workflow.add_node("extract_acknowledgements", extract_acknowledgements)

    # Define the edges/flow between nodes
    workflow.add_edge(START, "text_split")
    workflow.add_edge("text_split", "router_node")
    # Router -> Extractor Nodes (Conditional)
    workflow.add_conditional_edges(
        "router_node",
        lambda x: x,
        {
            "HeaderInformation": "extract_header_information",
            "AuthorList": "extract_author_list",
            "ScientificContent": "extract_scientific_content",
            "References": "extract_references",
            "ContactInformation": "extract_contact_information",
            "Acknowledgements": "extract_acknowledgements",
            "end_loop": END # Map the END constant returned by router
        }
    )
    # Extractor Nodes -> Router (Loop back)
    workflow.add_edge("extract_header_information", "router_node")
    workflow.add_edge("extract_author_list", "router_node")
    workflow.add_edge("extract_scientific_content", "router_node")
    workflow.add_edge("extract_references", "router_node")
    workflow.add_edge("extract_contact_information", "router_node")
    workflow.add_edge("extract_acknowledgements", "router_node")

    return workflow.compile()