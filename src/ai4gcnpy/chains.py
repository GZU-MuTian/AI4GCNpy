DEFAULT_TAGGING_PROMPT_TEMPLATE = """
You are an expert astronomer analyzing NASA GCN Circulars.

**Task:** Assign exactly ONE specific topic tag to each of the numbered paragraphs provided below.

**Allowed Tags (Choose Only From These):**
- HeaderInformation
- AuthorList
- ScientificContent
- References
- ContactInformation
- Acknowledgements

**Important Instructions:**
1.  **GCN Structure:** GCNs typically follow this structure:
    - 1st Paragraph: Usually `HeaderInformation` (containing title, date, from, subject).
    - 2nd Paragraph: Usually `AuthorList`.
    - Middle Paragraph(s): Primarily `ScientificContent`.
    - Last Paragraph(s) (if present): May contain `References`, `ContactInformation`, or `Acknowledgements`.
2.  **Input Format:** The paragraphs are prefixed with **P<N>**, where N is the paragraph's order (1, 2, 3, ...). This numbering is for your reference to assign the correct tag based on position and content. Do NOT use any numbers found WITHIN the paragraph text to influence your decision.
3.  **Output Format:** Respond ONLY with a valid JSON array of strings. Each string must be one of the allowed tags, corresponding to P1, P2, P3, ... in order.
    Example for 3 paragraphs: `["HeaderInformation", "AuthorList", "ScientificContent"]`

**Numbered Paragraphs:**
{numbered_paragraphs}
"""

 