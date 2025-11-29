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
    email = ""
    email_match  = re.fullmatch(r'\s*(.*?)\s*<([^>]+)>\s*', from_field)
    if email_match:
        submitter = email_match.group(1)
        email = email_match.group(2).strip()
    else:
        submitter = from_field

    return {
        "circularId": number.strip(),
        "subject": subject.strip(),
        "createdOn": date.strip(),
        "submitter": submitter.strip(),
        "email": email
    }