from .chains import TopicLabelerChain
from .utils import split_text_into_paragraphs, group_paragraphs_by_labels

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
        description="Paragraphs assigned topic label"
    )
    # remaining_tags: List[str] = Field(
    #     default_factory=list,
    #     description="List of tags yet to be processed by extractors."
    # )
    # extracted_data: Dict[str, Any] = Field(
    #     default_factory=dict, 
    #     description="Dict storing extracted physical information."
    # )

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

    return {"paragraphs": labeled_paragraphs}


# --- Extractor Nodes ---

def extract_header_information(state: CircularState) -> Dict[str, Any]:
    """
    Simulates extraction of header information from the current paragraph.

    Args:
        state (CircularState): The current state containing 'grouped_paragraphs'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_data' key.
    """
    logging.info("Extractor 'Header Information' activated.")
    current_tag = state.remaining_tags[0]
    paragraph = state.grouped_paragraphs.get(current_tag, "")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"header_sample": "example"}
    logging.debug(f"Extracted Header Info: {extracted_info}")

    updated_extracted = state.extracted_data.copy()
    updated_extracted.update(extracted_info)
    # Remove the processed tag
    new_remaining = state.remaining_tags[1:]
    return {
        "extracted_data": updated_extracted,
        "remaining_tags": new_remaining
    }

def extract_author_list(state: CircularState) -> Dict[str, Any]:
    """
    Simulates extraction of author list information.

    Args:
        state (CircularState): The current state containing 'grouped_paragraphs'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_data' key.
    """
    logging.info("Extractor 'Author List' activated.")
    current_tag = state.remaining_tags[0]
    paragraph = state.grouped_paragraphs.get(current_tag, "")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"author_list_sample": "example"}
    logging.debug(f"Extracted Author List: {extracted_info}")

    updated_extracted = state.extracted_data.copy()
    updated_extracted.update(extracted_info)
    # Remove the processed tag
    new_remaining = state.remaining_tags[1:]
    return {
        "extracted_data": updated_extracted,
        "remaining_tags": new_remaining
    }

def extract_scientific_content(state: CircularState) -> Dict[str, Any]:
    """
    Simulates extraction of scientific content details.

    Args:
        state (CircularState): The current state containing 'grouped_paragraphs'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_data' key.
    """ 
    logging.info("Extractor 'Scientific Content' activated.")
    current_tag = state.remaining_tags[0]
    paragraph = state.grouped_paragraphs.get(current_tag, "")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"scientific_content_sample": "example"}
    logging.debug(f"Extracted Scientific Content: {extracted_info}")

    updated_extracted = state.extracted_data.copy()
    updated_extracted.update(extracted_info)
    # Remove the processed tag
    new_remaining = state.remaining_tags[1:]
    return {
        "extracted_data": updated_extracted,
        "remaining_tags": new_remaining
    }

def extract_references(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): The current state containing 'grouped_paragraphs'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_data' key.
    """ 
    logging.info("Extractor 'References' activated.")
    current_tag = state.remaining_tags[0]
    paragraph = state.grouped_paragraphs.get(current_tag, "")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"references_sample": "example"}
    logging.debug(f"Extracted References: {extracted_info}")

    updated_extracted = state.extracted_data.copy()
    updated_extracted.update(extracted_info)
    # Remove the processed tag
    new_remaining = state.remaining_tags[1:]
    return {
        "extracted_data": updated_extracted,
        "remaining_tags": new_remaining
    }

def extract_contact_information(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): The current state containing 'grouped_paragraphs'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_data' key.
    """ 
    logging.info("Extractor 'Contact Information' activated.")
    current_tag = state.remaining_tags[0]
    paragraph = state.grouped_paragraphs.get(current_tag, "")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"contact_information_sample": "example"}
    logging.debug(f"Extracted Contact Information: {extracted_info}")

    updated_extracted = state.extracted_data.copy()
    updated_extracted.update(extracted_info)
    # Remove the processed tag
    new_remaining = state.remaining_tags[1:]
    return {
        "extracted_data": updated_extracted,
        "remaining_tags": new_remaining
    }

def extract_acknowledgements(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): The current state containing 'grouped_paragraphs'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_data' key.
    """ 
    logging.info("Extractor 'Acknowledgements' activated.")
    current_tag = state.remaining_tags[0]
    paragraph = state.grouped_paragraphs.get(current_tag, "")

    # --- Placeholder Extraction Logic ---
    extracted_info = {"acknowledgements_sample": "example"}
    logging.debug(f"Extracted Acknowledgements: {extracted_info}")

    updated_extracted = state.extracted_data.copy()
    updated_extracted.update(extracted_info)
    # Remove the processed tag
    new_remaining = state.remaining_tags[1:]
    return {
        "extracted_data": updated_extracted,
        "remaining_tags": new_remaining
    }

def router_node(state: CircularState) -> Dict[str, Any]:
    # Returning an empty dict signifies no direct state change by this node function.
    return {}

def route_to_extractor(state: CircularState) -> str:
    """
    Routes to the appropriate extractor based on the next tag in remaining_tags.    
    
    Args:
        state (CircularState): The current workflow state.
        
    Returns:
       str: Name of next node (e.g., 'extract_header_information') or 'end'.
    """
    if not state.remaining_tags:
        logging.info("No more tags to process. Ending workflow.")
        return "end"
    
    next_tag = state.remaining_tags[0]
    return next_tag

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
    # workflow.add_edge("group_paragraphs", "router_node")

    # Router -> Extractor Nodes (Conditional)
    workflow.add_conditional_edges(
        "router_node",
        route_to_extractor, # The function that decides
        {
            "HeaderInformation": "extract_header_information",
            "AuthorList": "extract_author_list",
            "ScientificContent": "extract_scientific_content",
            "References": "extract_references",
            "ContactInformation": "extract_contact_information",
            "Acknowledgements": "extract_acknowledgements",
            "end": END # Map the END constant returned by router
        }
    )
    # Extractor Nodes -> Router (Loop back)
    # After any extraction, go back to the router to check the next key
    workflow.add_edge("extract_header_information", "router_node")
    workflow.add_edge("extract_author_list", "router_node")
    workflow.add_edge("extract_scientific_content", "router_node")
    workflow.add_edge("extract_references", "router_node")
    workflow.add_edge("extract_contact_information", "router_node")
    workflow.add_edge("extract_acknowledgements", "router_node")

    return workflow.compile()