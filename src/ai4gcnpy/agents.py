from .prompts import DEFAULT_TAGGING_PROMPT_TEMPLATE

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import partial

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
import json
import logging


# --- State Definition ---

class GCNState(BaseModel):
    """Represents the state of the GCN tagging workflow."""
    raw_text: str = Field(..., description="Original GCN circular text.")
    paragraphs: List[str] = Field(
        default_factory=list,
        description="List of input GCN text paragraphs."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Corresponding topic tags for each paragraph."
    )
    grouped_paragraphs: Dict[str, str] = Field(
        default_factory=dict,
        description="Grouped paragraphs by category tags."
    )
    remaining_tags: List[str] = Field(
        default_factory=list,
        description="List of tags yet to be processed by extractors."
    )
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Dict storing extracted physical information."
    )

# --- Node Functions ---

def text_split(state: GCNState) -> Dict[str, Any]:
    """
    Loads the GCN text and splits it into paragraphs.
    
    Args:
        state (GCNState): The current state containing raw_text.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'paragraphs' key.
    """
    logging.info("Splitting GCN text into paragraphs.")
    raw_text = state.raw_text
    # Simple split by double newline.
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    logging.debug(f"Split into {len(paragraphs)} paragraphs.")
    return {"paragraphs": paragraphs}

def label_paragraphs(
        state: GCNState, 
        llm: Runnable, 
        PROMPT_TEMPLATE: str = DEFAULT_TAGGING_PROMPT_TEMPLATE
    ) -> Dict[str, Any]:
    """
    Assigns a topic label to each paragraph using an LLM.
    
    Args:
        state (GCNState): The current state containing 'paragraphs'.
        llm (Runnable): An LLM to use for tagging.
        PROMPT_TEMPLATE (str): An optional custom prompt template string. If empty, the default is used.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'tags' key.
    """
    logging.info("Labeling paragraphs with topics.")
    paragraphs = state.paragraphs

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE
    )
    tagging_chain = prompt | llm | StrOutputParser()    

    # Prepare input data with clear prefix P<N>
    numbered_paragraphs_parts = [f"P{i+1}: {p}" for i, p in enumerate(paragraphs)]
    numbered_paragraphs_str = "\n".join(numbered_paragraphs_parts)

    # LLM Integration Placeholder
    logging.debug("Invoking LLM for batch tagging...")
    response = tagging_chain.invoke({"numbered_paragraphs": numbered_paragraphs_str})
    logging.debug(f"tagging_chain Response:\n{response}")
    try:
        tags = json.loads(response)
    except Exception as e:
        logging.error(f"Error during LLM tagging process: {e}", exc_info=True)
        tags = []
    
    return {"tags": tags}

def group_paragraphs(state: GCNState) -> Dict[str, Any]:
    """
    Groups paragraphs based on their assigned labels.
    
    Args:
        state (GCNState): The current state containing 'paragraphs' and 'tags'.
        
    Returns:
        Dict[str, Any]: Updates the state with the 'grouped_paragraphs' key.
    """
    logging.info("Grouping paragraphs by topic labels.")
    paragraphs = state.paragraphs
    tags = state.tags
    grouped = {}

    if len(paragraphs) != len(tags):
        logging.error("Mismatch between number of paragraphs and tags. Cannot group.")

    for paragraph, tag in zip(paragraphs, tags):
        if tag in grouped:
            grouped[tag] += "\n\n" + paragraph
        else:
            grouped[tag] = paragraph

    remaining_tags = list(grouped.keys())
    logging.info(f"Paragraphs grouped into tags:\n{remaining_tags}")
    return {"grouped_paragraphs": grouped, "remaining_tags": remaining_tags}

# --- Extractor Nodes ---

def extract_header_information(state: GCNState) -> Dict[str, Any]:
    """
    Simulates extraction of header information from the current paragraph.

    Args:
        state (GCNState): The current state containing 'grouped_paragraphs'.
        
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

def extract_author_list(state: GCNState) -> Dict[str, Any]:
    """
    Simulates extraction of author list information.

    Args:
        state (GCNState): The current state containing 'grouped_paragraphs'.
        
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

def extract_scientific_content(state: GCNState) -> Dict[str, Any]:
    """
    Simulates extraction of scientific content details.

    Args:
        state (GCNState): The current state containing 'grouped_paragraphs'.
        
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

def extract_references(state: GCNState) -> Dict[str, Any]:
    """
    Args:
        state (GCNState): The current state containing 'grouped_paragraphs'.
        
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

def extract_contact_information(state: GCNState) -> Dict[str, Any]:
    """
    Args:
        state (GCNState): The current state containing 'grouped_paragraphs'.
        
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

def extract_acknowledgements(state: GCNState) -> Dict[str, Any]:
    """
    Args:
        state (GCNState): The current state containing 'grouped_paragraphs'.
        
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

def router_node(state: GCNState) -> Dict[str, Any]:
    # Returning an empty dict signifies no direct state change by this node function.
    return {}

def route_to_extractor(state: GCNState) -> str:
    """
    Routes to the appropriate extractor based on the next tag in remaining_tags.    
    
    Args:
        state (GCNState): The current workflow state.
        
    Returns:
       str: Name of next node (e.g., 'extract_header_information') or 'end'.
    """
    if not state.remaining_tags:
        logging.info("No more tags to process. Ending workflow.")
        return "end"
    
    next_tag = state.remaining_tags[0]
    return next_tag

# --- Graph Construction ---

def GCNParserGraph(llm: Runnable):
    """
    Creates and compiles the LangGraph workflow for GCN processing.

    Args:
        llm (Runnable): An LLM to use for tagging.

    Returns:
        StateGraph: The compiled workflow graph.
    """
    logging.info("Creating LangGraph workflow.")
    # Initialize the state graph 
    workflow = StateGraph(GCNState)

    # Add nodes
    workflow.add_node("text_split", text_split)
    workflow.add_node(
        "label_paragraphs", 
        partial(label_paragraphs, llm=llm, PROMPT_TEMPLATE=DEFAULT_TAGGING_PROMPT_TEMPLATE)
    )
    workflow.add_node("group_paragraphs", group_paragraphs)
    workflow.add_node("router_node", router_node)
    workflow.add_node("extract_header_information", extract_header_information)
    workflow.add_node("extract_author_list", extract_author_list)
    workflow.add_node("extract_scientific_content", extract_scientific_content)
    workflow.add_node("extract_references", extract_references)
    workflow.add_node("extract_contact_information", extract_contact_information)
    workflow.add_node("extract_acknowledgements", extract_acknowledgements)

    # Define the edges/flow between nodes
    workflow.add_edge(START, "text_split")
    workflow.add_edge("text_split", "label_paragraphs")
    workflow.add_edge("label_paragraphs", "group_paragraphs")
    workflow.add_edge("group_paragraphs", "router_node")

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

    return workflow