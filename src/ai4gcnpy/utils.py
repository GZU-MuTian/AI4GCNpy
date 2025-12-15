import re
from typing import List, Dict, Tuple, Any, Optional
import logging
from datetime import date

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


def build_cypher_statements(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate a list of (Cypher query, parameters) tuples from validated circular data.

    Args:
        data (Dict[str, Any]): Validated input dictionary.

    Returns:
        List[Tuple[str, Dict[str, Any]]]: List of executable Cypher statement-parameter pairs.
    """
    statements: List[Tuple[str, Dict[str, Any]]] = []

    dset: dict = data.get("extracted_dset", {})

    # 1. Create CIRCULAR node
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

    # 2. COLLABORATION（仅当 collaboration 非空）
    # collaboration = d.get("collaboration")
    # if collaboration:
    #     q2 = """
    #     MATCH (c:CIRCULAR {circularId: $circularId})
    #     CREATE (collab:COLLABORATION {name: $collaborationName})
    #     CREATE (c)-[:REPORT]->(collab)
    #     """
    #     p2 = {"circularId": circular_id, "collaborationName": collaboration}
    #     statements.append((q2, p2))

    # 3. AUTHORS（仅当 authors 存在且非空）
    # authors_input = d.get("authors")
    # if authors_input:
    #     try:
    #         authors_list = [
    #             {"name": a["author"], "affiliation": a["affiliation"]}
    #             for a in authors_input
    #             if a.get("author") and a.get("affiliation")
    #         ]
    #         if authors_list:
    #             q3 = """
    #             MATCH (c:CIRCULAR {circularId: $circularId})
    #             MATCH (collab:COLLABORATION {name: $collaborationName})
    #             UNWIND $authors AS auth
    #             MERGE (a:AUTHOR {name: auth.name})
    #             MERGE (inst:INSTITUTION {name: auth.affiliation})
    #             MERGE (a)-[:AFFILIATED_WITH]->(inst)
    #             MERGE (a)-[:MEMBER_OF]->(collab)
    #             MERGE (c)-[:HAS_AUTHOR]->(a)
    #             """
    #             p3 = {
    #                 "circularId": circular_id,
    #                 "collaborationName": collaboration or "",  # 即使无 collaboration 也允许（MERGE 不依赖）
    #                 "authors": authors_list
    #             }
    #             statements.append((q3, p3))
    #     except (TypeError, KeyError):
    #         pass  # 忽略格式错误的 authors

    # 4. INTENT（仅当 intent 非空）
    # intent_type = d.get("intent")
    # if intent_type:
    #     q4 = """
    #     MATCH (c:CIRCULAR {circularId: $circularId})
    #     CREATE (intent:INTENT {name: $intentType})
    #     CREATE (c)-[:HAS_INTENT]->(intent)
    #     """
    #     p4 = {"circularId": circular_id, "intentType": intent_type}
    #     statements.append((q4, p4))

    # 5. PHYSICAL_QUANTITY（仅当至少一个物理量非 None）
    # quantity_fields = [
    #     "position_and_coordinates",
    #     "time_and_duration",
    #     "flux_and_brightness",
    #     "spectrum_and_energy",
    #     "observation_conditions_and_instrument",
    #     "distance_and_redshift",
    #     "extinction_and_absorption",
    #     "statistical_significance_and_uncertainty",
    #     "upper_limit",
    #     "source_identification_and_characteristics"
    # ]

    # quantity_list = []
    # for field in quantity_fields:
    #     sentences = d.get(field)
    #     if sentences is not None:  # 显式非 None（允许空列表，但通常不会）
    #         quantity_list.append({"type": field, "sentences": sentences})

    # if quantity_list:
    #     q5 = """
    #     MATCH (c:CIRCULAR {circularId: $circularId})
    #     UNWIND $quantityList AS qty
    #     FOREACH (_ IN CASE WHEN qty.sentences IS NOT NULL THEN [1] ELSE [] END |
    #       CREATE (pq:PHYSICAL_QUANTITY {type: qty.type})
    #       CREATE (c)-[:MENTIONS {sentences: qty.sentences}]->(pq)
    #     )
    #     """
    #     p5 = {"circularId": circular_id, "quantityList": quantity_list}
    #     statements.append((q5, p5))

    return statements