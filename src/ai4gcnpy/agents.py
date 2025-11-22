from .prompts import DEFAULT_TAGGING_PROMPT_TEMPLATE

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import partial

from typing import List, Dict, Any
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

    logging.debug(f"Paragraphs grouped into tags: {list(grouped.keys())}")
    return {"grouped_paragraphs": grouped}

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

    # Define the edges/flow between nodes
    workflow.add_edge(START, "text_split")
    workflow.add_edge("text_split", "label_paragraphs")
    workflow.add_edge("label_paragraphs", "group_paragraphs")
    workflow.add_edge("group_paragraphs", END)

    return workflow