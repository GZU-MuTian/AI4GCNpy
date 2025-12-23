from .chains import ParagraphLabelerChain, ParseAuthorshipChain, ReportLabelerChain, PhysicalQuantityExtractorChain, GuardrailsChain, Text2CypherChain, GenerateFinalChain, ALLOWED_PARAGRAPH_LABELS
from .chains import ParagraphLabelList, AuthorList, ReportLabel, PhysicalQuantityCategory
from .utils import split_text_into_paragraphs, group_paragraphs_by_labels, header_regex_match, extract_cypher

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

    extracted_info = {}
    # Label using LLM
    try:  
        label_chain = ReportLabelerChain()
        label_responses: ReportLabel = label_chain.invoke({"content": paragraph})
        extracted_info.update({"intent": label_responses.label})
    except Exception as e:
        logger.error(f"ReportLabelerChain | Failed to label topic: {e}")
        raise

    # extract using LLM
    try:  
        quantity_chain = PhysicalQuantityExtractorChain()
        quantity_responses: PhysicalQuantityCategory = quantity_chain.invoke({"content": paragraph})
        extracted_info.update(quantity_responses.model_dump())
    except Exception as e:
        logger.error(f"ReportLabelerChain | Failed to label topic: {e}")
        raise

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



# --- State Definition ---

class GraphQAState(BaseModel):
    """
    Represents the state passed through the QA workflow graph.
    """
    query: str = Field(..., description="The original user question.")
    graph: Any = Field(..., description="Graph database interface with get_schema() method.")
    database: Optional[str] = Field(None, description="Target database name (optional).")
    cypher_statement: str = Field("", description="Generated Cypher query.")
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw records returned from graph DB execution."
    )
    answer: str = Field("", description="Final natural language answer in Markdown.")
    next_action: Optional[str] = Field(None, description="Next node to route to.")

# --- Node Functions ---

def guardrails(state: GraphQAState) -> Dict[str, Any]:
    """
    Uses an LLM to decide if the question is related to NASA's GCN.
    """  
    try:
        guardrails_chain = GuardrailsChain()
        guardrails_output = guardrails_chain.invoke({
            "question": state.query,
            "schema": state.graph.get_schema()
        })

        if guardrails_output.decision == "gcn":
            return {
                "next_action": "generate_cypher"
            }
        else:
            return {
                "answer": "I specialize exclusively in NASA's General Coordinates Network (GCN). Your question appears unrelated to this domain, so I cannot assist.",
                "next_action": "end",
            }
    except:
        return {
            "answer": "I'm currently unable to determine if your question relates to NASA's General Coordinates Network (GCN). Please try rephrasing your query.",
            "next_action": "end",
        }

def generate_cypher(state: GraphQAState) -> Dict[str, Any]:
    """
    Converts query_text to a Cypher query using an LLM.

    Args:
        state: Current workflow state containing the user query.
        
    Returns:
        Updated state with a (placeholder) Cypher statement.
    """
    try:
        cypher_chain = Text2CypherChain()
        cypher_statement = cypher_chain.invoke({
            "question": state.query, 
            "schema": state.graph.get_schema()
        })
        cypher_query = extract_cypher(cypher_statement)
        logger.debug(f"Generated Cypher: {cypher_query}")
        return {
            "cypher_statement": cypher_query,
            "next_action": "execute_cypher",
        }
    except:
        return {
            "answer": "Failed to generate Cypher. Please try rephrasing your query.",
            "next_action": "end",
        }

def execute_cypher(state: GraphQAState) -> Dict[str, Any]:
    """
    Executes the validated Cypher query against a graph database.
        
    Args:
        state: Current state with a valid Cypher statement.
        
    Returns:
        Updated state with retrieved result chunks.
    """
    cypher_statement = state.cypher_statement
    database = state.database
    graph = state.graph

    try:
        with graph.session(database) as session:
            result_cursor = session.run(cypher_statement)
            retrieved_chunks = [record.data() for record in result_cursor]
        return {
            "retrieved_chunks": retrieved_chunks,
            "next_action": "generate_final_answer"
        }
    except:
        return {
            "answer": "Cypher execution failed. Please try simplifying your question or rephrase your request.",
            "next_action": "end"
        }

def generate_final_answer(state: GraphQAState) -> Dict[str, Any]:
    """
    Generates a natural language answer from the retrieved graph data.
        
    Args:
        state: Current state with retrieved data.
        
    Returns:
        Final state with a natural language answer.
    """
    question = state.query
    retrieved_chunks = state.retrieved_chunks

    # Convert retrieved records to a readable string context
    if not retrieved_chunks:
        context_str = "No relevant data found in the knowledge graph."
    else:
        # Serialize each record into a readable format (e.g., JSON-like but human-friendly)
        context_lines = []
        for idx, record in enumerate(retrieved_chunks, start=1):
            line = f"Record {idx}: {record}"
            context_lines.append(line)
        context_str = "\n".join(context_lines)

    try:
        generate_final_chain = GenerateFinalChain()
        answer = generate_final_chain.invoke({
            "question": question,
            "results": context_str
        })
        return {
            "answer": answer.strip(),
            "next_action": "end"
        }
    except Exception as e:
        logger.error(f"Error in generate_final_answer: {e}")
        return {
            "answer": "I encountered an error while generating the final response. Please try again.",
            "next_action": "end"
        }


# --- Graph Construction ---

def GraphQAAgent():

    workflow = StateGraph(GraphQAState)

    workflow.add_node("guardrails", guardrails)
    workflow.add_node("generate_cypher", generate_cypher)
    workflow.add_node("execute_cypher", execute_cypher)
    workflow.add_node("generate_final_answer", generate_final_answer)

    # Define the edges/flow between nodes
    workflow.add_edge(START, "guardrails")
    workflow.add_conditional_edges(
        "guardrails",
        lambda state: state.next_action,
        {
            "generate_cypher": "generate_cypher",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "generate_cypher",
        lambda state: state.next_action,
        {
            "execute_cypher": "execute_cypher",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "execute_cypher",
        lambda state: state.next_action,
        {
            "generate_final_answer": "generate_final_answer",
            "end": END,
        },
    )
    workflow.add_edge("generate_final_answer", END)

    return workflow.compile()