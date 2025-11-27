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