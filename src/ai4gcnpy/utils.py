import re
from typing import List, Dict, Tuple, Any, Optional, LiteralString
import logging
from datetime import date
import tempfile
import tarfile
from pathlib import Path
import urllib.request


logger = logging.getLogger(__name__)


def split_text_into_paragraphs(raw_text: str) -> List[str]:
    """
    Splits raw GCN text into a list of non-empty paragraphs using double newline as delimiter.

    Args:
        raw_text (str): The full input text.

    Returns:
        List[str]: A list of stripped, non-empty paragraphs.
    """
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', raw_text)
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
    logger.debug(f"Split text into {len(paragraphs)} paragraphs.")
    return paragraphs

def group_paragraphs_by_labels(
    paragraphs: List[str], tags: List[str]
) -> Dict[str, str]:
    """
    Groups paragraphs by their corresponding topic labels.

    Args:
        paragraphs (List[str]): Original list of paragraphs.
        tags (List[str]): Topic label for each paragraph (must align 1:1).

    Returns:
        Dict[str, str]: Mapping from tag to concatenated paragraph content.
    """
    if len(paragraphs) != len(tags):
        raise ValueError(
            f"Paragraph-tag length mismatch: {len(paragraphs)} vs {len(tags)}"
        )

    grouped: Dict[str, str] = {}
    for para, tag in zip(paragraphs, tags):
        if tag in grouped:
            grouped[tag] += "\n\n" + para
        else:
            grouped[tag] = para

    logger.debug(f"Grouped paragraphs into {len(grouped)} topics: {list(grouped.keys())}")
    return grouped


def header_regex_match(header: str) -> Dict[str, Any]:
    """
    Parses the header of a GCN circular using regex and returns a validated Pydantic model instance.

    Args:
        header (str): The raw header text of the GCN circular.

    Returns:
        Dict[str, Any]: A validated dict containing parsed metadata.
    """
    # Define expected header structure with regex (VERBOSE for readability)
    pattern = re.compile(r"""
        TITLE:\s*(.*?)\s*
        NUMBER:\s*(.*?)\s*
        SUBJECT:\s*(.*?)\s*
        DATE:\s*(.*?)\s*
        FROM:\s*(.*?)(?:\s*\n|$)
    """, re.VERBOSE)

    # match check
    match = pattern.search(header)
    if not match:
        raise ValueError("Header does not match expected GCN Circular format.")

    title, number, subject, date, from_field = match.groups()

    # Try to extract email
    submitter_match  = re.fullmatch(r'\s*(.*?)\s*<([^>]+)>\s*', from_field)
    if submitter_match:
        submitter = submitter_match.group(1).strip()
        email = submitter_match.group(2).strip()
    else:
        submitter = from_field.strip()
        email = ""

    return {
        "circularId": number.strip(),
        "subject": subject.strip(),
        "createdOn": date.strip(),
        "submitter": submitter,
        "email": email
    }


def build_cypher_statements(data: Dict[str, Any]) -> List[Tuple[LiteralString, Dict[str, Any]]]:
    """
    Generate a list of (Cypher query, parameters) tuples from validated circular data.

    Args:
        data (Dict[str, Any]): Validated input dictionary.

    Returns:
        List[Tuple[str, Dict[str, Any]]]: List of executable Cypher statement-parameter pairs.
    """
    statements: List[Tuple[LiteralString, Dict[str, Any]]] = []

    dset: dict = data.get("extracted_dset", {})

    # 1. Handle CIRCULAR node
    circular_node = """
        CREATE (c:CIRCULAR {
            circularId: $circularId,
            subject: $subject,
            createdOn: $createdOn,
            submitter: $submitter,
            email: $email,
            rawText: $rawText,
            ingestedBy: $ingestedBy,
            ingestedAt: $ingestedAt
        })
    """
    circular_para = {
        "circularId": dset.get("circularId"),
        "subject": dset.get("subject"),
        "createdOn": dset.get("createdOn"),
        "submitter": dset.get("submitter"),
        "email": dset.get("email"),
        "rawText": data.get("raw_text"),
        "ingestedBy": "AI4GCNpy",
        "ingestedAt": date.today()
    }
    statements.append((circular_node, circular_para))

    # --- Handle COLLABORATION node (if collaboration is non-empty) ---
    collaboration: str = dset.get("collaboration", "")
    if collaboration and collaboration.lower() != "null":
        collab_node = """
            MATCH (c:CIRCULAR {circularId: $circularId})
            MERGE (collab:COLLABORATION {name: $collaborationName})
            ON CREATE SET
                collab.ingestedBy = $ingestedBy,
                collab.ingestedAt = $ingestedAt
            CREATE (collab)-[:REPORT {
                ingestedBy: $ingestedBy,
                ingestedAt: $ingestedAt
            }]->(c)
        """
        collab_para = {
            "circularId": dset.get("circularId"), 
            "collaborationName": collaboration,
            "ingestedBy": "AI4GCNpy",
            "ingestedAt": date.today()
        }
        statements.append((collab_node, collab_para))

    # --- Handle AUTHOR node (if AUTHOR is non-empty) ---
    authors = dset.get("authors", [])
    if authors:
        author_node = """
            MATCH (c:CIRCULAR {circularId: $circularId})
            OPTIONAL MATCH (collab:COLLABORATION {name: $collaborationName})
            UNWIND $authors AS auth
            MERGE (a:AUTHOR {
                name: auth.author,
                affiliation: auth.affiliation
            })
            ON CREATE SET
                a.ingestedBy = $ingestedBy,
                a.ingestedAt = $ingestedAt
            FOREACH (_ IN CASE WHEN collab IS NOT NULL THEN [1] ELSE [] END |
                MERGE (a)-[:MEMBER_OF]->(collab)
            )
            MERGE (c)-[:HAS_AUTHOR]->(a)
        """
        author_para = {
            "circularId": dset.get("circularId"),
            "collaborationName": collaboration,
            "authors": authors,
            "ingestedBy": "AI4GCNpy",
            "ingestedAt": date.today()
        }
        statements.append((author_node, author_para))


    # --- Handle INTENT node (if intent is non-empty) ---
    intent_type = dset.get("intent")
    if intent_type:
        intent_node = """
        MATCH (c:CIRCULAR {circularId: $circularId})
        MERGE (intent:INTENT {name: $intentType})
        ON CREATE SET
            intent.ingestedBy = $ingestedBy
        CREATE (c)-[:HAS_INTENT {
            ingestedBy: $ingestedBy,
            ingestedAt: $ingestedAt
        }]->(intent)
        """
        intent_para = {
            "circularId": dset.get("circularId"), 
            "intentType": intent_type,
            "ingestedBy": "AI4GCNpy",
            "ingestedAt": date.today()
        }
        statements.append((intent_node, intent_para))

    # --- Handle PHYSICAL_QUANTITY node (if intent is non-empty) ---
    for field in [
        "position_and_coordinates",
        "time_and_duration",
        "flux_and_brightness",
        "spectrum_and_energy",
        "observation_conditions_and_instrument",
        "distance_and_redshift",
        "extinction_and_absorption",
        "statistical_significance_and_uncertainty",
        "upper_limit",
        "source_identification_and_characteristics"
    ]:
        sentences = dset.get(field)
        if not sentences:
            continue
        physical_quantity_node = """
            MATCH (c:CIRCULAR {circularId: $circularId})
            MERGE (pq:PHYSICAL_QUANTITY {name: $quantityName})
            ON CREATE SET
                pq.ingestedBy = $ingestedBy
            MERGE (c)-[:HAS_PHYSICAL_QUANTITY {
                sentences: $sentences,
                ingestedBy: $ingestedBy,
                ingestedAt: $ingestedAt
            }]->(pq)
        """
        physical_quantity_para = {
            "circularId": dset.get("circularId"),
            "quantityName": field,
            "sentences": sentences,
            "ingestedBy": "AI4GCNpy",
            "ingestedAt": date.today()
        }
        statements.append((physical_quantity_node, physical_quantity_para))

    # --- Handle METADATA node (if intent is non-empty) ---
    for field in [
        "externalLinks",
        "contactInformation",
        "acknowledgements",
        "citationInstructions",
        "correction",
        "unknown"
    ]:
        sentences = dset.get(field)
        if not sentences:
            continue
        metadata_node = """
            MATCH (c:CIRCULAR {circularId: $circularId})
            MERGE (m:METADATA {name: $metadateType})
            ON CREATE SET
                m.ingestedBy = $ingestedBy
            MERGE (c)-[:HAS_METADATA {
                sentences: $sentences,
                ingestedBy: $ingestedBy,
                ingestedAt: $ingestedAt
            }]->(m)
        """
        metadata_para = {
            "circularId": dset.get("circularId"),
            "metadateType": field,
            "sentences": sentences,
            "ingestedBy": "AI4GCNpy",
            "ingestedAt": date.today()
        }
        statements.append((metadata_node, metadata_para))

    return statements


def extract_cypher(text: str) -> str:
    """Extract and format Cypher query from text, handling code blocks and special characters. See neo4j_graphrag.retrievers.text2cypher.

    This function performs two main operations:
    1. Extracts Cypher code from within triple backticks (```), if present
    2. Automatically adds backtick quotes around multi-word identifiers:
       - Node labels (e.g., ":Data Science" becomes ":`Data Science`")
       - Property keys (e.g., "first name:" becomes "`first name`:")
       - Relationship types (e.g., "[:WORKS WITH]" becomes "[:`WORKS WITH`]")

    Args:
        text (str): Raw text that may contain Cypher code, either within triple
                   backticks or as plain text.

    Returns:
        str: Properly formatted Cypher query with correct backtick quoting.
    """
    # Extract Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    cypher_query = matches[0] if matches else text
    # Quote node labels in backticks if they contain spaces and are not already quoted
    cypher_query = re.sub(
        r":\s*(?!`\s*)(\s*)([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)+)(?!\s*`)(\s*)",
        r":`\2`",
        cypher_query,
    )
    # Quote property keys in backticks if they contain spaces and are not already quoted
    cypher_query = re.sub(
        r"([,{]\s*)(?!`)([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)+)(?!`)(\s*:)",
        r"\1`\2`\3",
        cypher_query,
    )
    # Quote relationship types in backticks if they contain spaces and are not already quoted
    cypher_query = re.sub(
        r"(\[\s*[a-zA-Z0-9_]*\s*:\s*)(?!`)([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)+)(?!`)(\s*(?:\]|-))",
        r"\1`\2`\3",
        cypher_query,
    )
    return cypher_query

def progress_bar(block_num, block_size, total_size):
    if total_size > 0:
        downloaded = block_num * block_size
        percent = downloaded * 100 / total_size
        
        # 创建进度条
        bar_length = 30
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)

def download_gcn_archive(url: str) -> str:
    """
    Download a .tar.gz archive containing GCN Circular TXT files from the given URL, save it to a temporary directory, extract its contents, and return the path to the extracted directory.

    Args:
        url (str): The URL pointing to a .tar.gz file containing GCN .txt files.

    Returns:
        str: The path to the directory where the archive was extracted.
    """
    # Create a temporary directory that persists until caller cleans up or program exits
    temp_dir = Path(tempfile.mkdtemp(prefix="gcn_extractor_"))
    archive_path = temp_dir / "archive.txt.tar.gz"
    
    logger.debug(f"Downloading GCN archive from: {url}")
    try:
        urllib.request.urlretrieve(url, archive_path, reporthook=progress_bar)
    except:
        raise RuntimeError(f"Failed to download archive from {url}.")

    logger.info(f"Extracting .tar.gz archive to: {temp_dir}")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_dir, filter='data')
    except:
        raise RuntimeError(f"Failed to extract .tar.gz archive")
    
    extracted_dir = temp_dir / "archive.txt"
    return str(extracted_dir.resolve())