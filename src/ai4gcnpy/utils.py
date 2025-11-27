import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def split_text_into_paragraphs(raw_text: str) -> List[str]:
    """
    Splits raw GCN text into a list of non-empty paragraphs using double newline as delimiter.

    Args:
        raw_text (str): The full input text.

    Returns:
        List[str]: A list of stripped, non-empty paragraphs.
    """
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
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


def header_regex_match(header: str) -> dict:
    """
    Parses the header of a GCN circular

    Args:
        header (str): The header of the GCN circular
    
    Returns:
        dict: A dictionary containing the parsed header information
    """
    # metadata structure
    metadata = {
        "circularId": '',
        "subject": '',
        "createdOn": '',
        "submitter": '',
        "email": '',
    }

    # Regular expression pattern to match the header
    pattern = re.compile(r"""
        TITLE:\s*(.*?)\s*
        NUMBER:\s*(.*?)\s*
        SUBJECT:\s*(.*?)\s*
        DATE:\s*(.*?)\s*
        FROM:\s*(.*?)(?:\s*\n|$)
    """, re.VERBOSE)
    match = pattern.search(header)

    # match check
    if not match or match.group(1) != 'GCN CIRCULAR':
        logging.debug(f"Failed to parse document:\n{header}")
        return metadata

    # metadata structure
    metadata.update({
        "circularId": match.group(2),
        "subject": match.group(3),
        "createdOn": match.group(4),
        "submitter": match.group(5),
    })

    email_match = re.search(r'<([^>]+)>', match.group(5))
    if email_match:
        metadata.update({"email": email_match.group(1)})
    else:
        logging.debug(f"Failed to parse email from submitter: {match.group(5)}")

    return metadata