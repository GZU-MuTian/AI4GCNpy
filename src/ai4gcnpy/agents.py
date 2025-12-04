from .chains import ParagraphLabelerChain, ParseAuthorshipChain, ReportLabelerChain, ALLOWED_PARAGRAPH_LABELS, ALLOWED_REPORT_LABELS
from .utils import split_text_into_paragraphs, group_paragraphs_by_labels, header_regex_match
from .chains import ParagraphLabelList, AuthorList, ReportLabel

from langgraph.graph import StateGraph, START, END

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
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
    current_label: str = Field(default="end_loop", description="The label currently being processed.")

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
        raise ValueError("No paragraphs found in input text.")

    # Prepare input data with clear prefix P<N>
    numbered_paragraphs_parts = [f"<P{i+1}>{p}</P{i+1}>" for i, p in enumerate(paragraphs)]
    numbered_paragraphs_str = "\n\n".join(numbered_paragraphs_parts)
    logger.debug(f"Split paragraphs:\n{numbered_paragraphs_str}")

    # Label using LLM
    try:  
        chain = ParagraphLabelerChain()
        responses: ParagraphLabelList = chain.invoke({"numbered_paragraphs": numbered_paragraphs_str})
        labels = responses.labels
        logger.info(f"Paragraph labeling results: {labels}")
    except Exception as e:
        logger.error(f"ParagraphLabelerChain | Failed to label topic: {e}")
        raise

    labeled_paragraphs = group_paragraphs_by_labels(paragraphs, responses.labels)

    return {"paragraphs": labeled_paragraphs, "pending_labels": labeled_paragraphs.keys()}

def router_node(state: CircularState)  -> Dict[str, Any]:
    """
    Routing function that determines the next node to execute based on the current 'pending_labels' list.
    
    - If the 'pending_labels' list is empty, the workflow should end, so return 'end'.
    - Otherwise, return the first task in the 'pending_labels' list, which must match the name of a registered node.
    """
    pending_labels = state.pending_labels
    if not pending_labels:
        logger.debug("Router: No pending labels — exiting loop.")
        return {"current_label": "end_loop"} 
    
    current_label = pending_labels[0]
    logger.debug(f"Router: Selected next node '{current_label}'")
    return {"current_label": current_label}

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
    logger.debug("Successfully extracted head information: %s", extracted_info)

    # Update extracted dataset
    current_extracted = state.extracted_dset
    updated_extracted = {**current_extracted, **extracted_info}

    # Remove the processed label
    updated_pending = state.pending_labels[1:]
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
    # Remove the processed label
    updated_pending = state.pending_labels[1:]

    paragraph = state.paragraphs.get("AuthorList", "")
    if not paragraph.strip():
        logger.warning("AuthorList paragraph is empty or missing.")
        return {"pending_labels": updated_pending}

    # parse GCN Circular author list
    try:
        chain = ParseAuthorshipChain()
        responses: AuthorList = chain.invoke({"content": paragraph})
    except Exception as e:
        logger.error(f"Failed to parse author list: {e}")
        return {"pending_labels": updated_pending}

    # Update extracted dataset
    current_extracted = state.extracted_dset
    updated_extracted = {**current_extracted, **responses.model_dump()}

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
    # Remove the processed label
    updated_pending = state.pending_labels[1:]

    paragraph = state.paragraphs.get("ScientificContent", "")
    if not paragraph.strip():
        logger.warning("ScientificContent paragraph is empty or missing.")
        return {"pending_labels": updated_pending}

    # Label using LLM
    try:  
        label_chain = ReportLabelerChain()
        label_responses: ReportLabel = label_chain.invoke({"content": paragraph})
        label = label_responses.label
        logger.info(f"Primary Label: {label}")
    except Exception as e:
        logger.error(f"ReportLabelerChain | Failed to label topic: {e}")
        raise

    # number using LLM

    # --- Placeholder Extraction Logic ---
    extracted_info = {"intent": label}
    logger.debug("Successfully extracted information: %s", extracted_info)

    # Update extracted dataset
    current_extracted = state.extracted_dset
    updated_extracted = {**current_extracted, **extracted_info}


    return {
        "extracted_dset": updated_extracted,
        "pending_labels": updated_pending
    }


def retain_original_text(state: CircularState) -> Dict[str, Any]:
    """
    Args:
        state (CircularState): Current graph state containing 'paragraphs' and 'pending_labels'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'extracted_dset' key.
    """ 
    # Remove the processed label
    current_label = state.current_label
    updated_pending = state.pending_labels[1:]

    paragraph = state.paragraphs.get(current_label, "")
    if not paragraph.strip():
        logger.warning(f"{current_label} paragraph is empty or missing.")
        return {"pending_labels": updated_pending}

    key = current_label[:1].lower() + current_label[1:]
    extracted_info = {key: paragraph}

    # Update extracted dataset
    current_extracted = state.extracted_dset
    updated_extracted = {**current_extracted, **extracted_info}
    
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
    logging.debug("Creating GCNExtractorAgent workflow.")
    # Initialize the state graph 
    workflow = StateGraph(CircularState)

    # Add nodes
    workflow.add_node("text_split", text_split)
    workflow.add_node("router_node", router_node)
    workflow.add_node("extract_header_information", extract_header_information)
    workflow.add_node("extract_author_list", extract_author_list)
    workflow.add_node("extract_scientific_content", extract_scientific_content)
    workflow.add_node("retain_original_text", retain_original_text)

    # Define the edges/flow between nodes
    workflow.add_edge(START, "text_split")
    workflow.add_edge("text_split", "router_node")
    # Router -> Extractor Nodes (Conditional)
    workflow.add_conditional_edges(
        "router_node",
        lambda state: state.current_label if (state.current_label in ALLOWED_PARAGRAPH_LABELS or state.current_label == "end_loop") else "Unknown",
        {
            "HeaderInformation": "extract_header_information",
            "AuthorList": "extract_author_list",
            "ScientificContent": "extract_scientific_content",
            "ExternalLinks": "retain_original_text",
            "ContactInformation": "retain_original_text",
            "Acknowledgements": "retain_original_text",
            "CitationInstructions": "retain_original_text",
            "Correction": "retain_original_text",
            "Unknown": "retain_original_text", 
            "end_loop": END,
        },
    )
    # Extractor Nodes -> Router (Loop back)
    workflow.add_edge("extract_header_information", "router_node")
    workflow.add_edge("extract_author_list", "router_node")
    workflow.add_edge("extract_scientific_content", "router_node")
    workflow.add_edge("retain_original_text", "router_node")

    return workflow.compile()