from . import llm_client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import logging

logger = logging.getLogger(__name__)


# --- TopicLabelerChain ---

_ALLOWED_LABEL = Literal[
    "HeaderInformation",
    "AuthorList",
    "ScientificContent",
    "References",
    "ContactInformation",
    "Acknowledgements"
]
class LabelList(BaseModel):
    labels: List[_ALLOWED_LABEL] = Field(description="A list of allowed labels, one per paragraph in order.")

labels_parser = PydanticOutputParser(pydantic_object=LabelList)

_SYSTEM_LABEL_PROMPT = """
You are an expert astronomer analyzing NASA GCN Circulars.

**Task:** Assign exactly ONE specific topic Label to each of the numbered paragraphs provided below.

**Allowed topics (Choose Only From These):**
- HeaderInformation
- AuthorList
- ScientificContent
- References
- ContactInformation
- Acknowledgements

**Important Instructions:**
1.  GCNs typically follow this structure:
    - 1st Paragraph: Usually `HeaderInformation` (containing TITLE, NUMBER, SUBJECT, DATE, FROM).
    - 2nd Paragraph: Usually `AuthorList`.
    - Middle Paragraph(s): Primarily `ScientificContent`.
    - Last Paragraph(s) (if present): May contain `References`, `ContactInformation`, or `Acknowledgements`.
2.  Input Format: Each paragraph is enclosed in paired tags <PN>...</PN>, where N is the paragraph's order (1, 2, 3, ...). This numbering is for your reference to assign the correct tag based on position and content. Do NOT use any numbers found WITHIN the paragraph text to influence your decision.
3.  Output Format:
{format_instructions}
Example for 3 paragraphs: `["HeaderInformation", "AuthorList", "ScientificContent"]`
"""


_HUMAN_LABEL_PROMPT = """
**Numbered Paragraphs:**
{numbered_paragraphs}
"""

LABEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_LABEL_PROMPT),
    ("human", _HUMAN_LABEL_PROMPT)
]).partial(format_instructions=labels_parser.get_format_instructions())

def TopicLabelerChain():
    """
    Assign topic labels to paragraphs.
    """
    llm = llm_client.getLLM()
    return LABEL_PROMPT | llm | labels_parser

# --- TopicLabelerChain ---

class AuthorEntry(BaseModel):
    author: str = Field(description="Author name.")
    affiliation: str = Field(description="Institutional affiliation.")

class Collaboration(BaseModel):
    collaboration: Optional[str] = Field(default=None, description="Name of the collaboration or team, or null if not mentioned")
    authors: List[AuthorEntry] = Field(default_factory=list, description="List of authors and their affiliations")

authorship_parser = PydanticOutputParser(pydantic_object=Collaboration)

_SYSTEM_AUTHORSHIP_PROMPT = """
You are an expert in parsing astronomical and scientific authorship lists. Your task is to extract structured information from the input text.

**Instructions:**
1. The text contains one or more groups of authors followed by their institutional affiliations in parentheses.
2. All authors listed before a parenthetical institution belong to that institution.
3. Additionally, check if the text ends with a phrase like "report on behalf of the [Team Name] team" or similar. If so, record the team name.
4. Author names may appear as "Initial. Lastname". Preserve spacing and punctuation as given.

{format_instructions}
"""

_HUMAN_AUTHORSHIP_PROMPT = """
**Input Text:**
{content}
"""

AUTHORSHIP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_AUTHORSHIP_PROMPT),
    ("human", _HUMAN_AUTHORSHIP_PROMPT)
]).partial(format_instructions=authorship_parser.get_format_instructions())

def ParseAuthorshipChain():
    llm = llm_client.getLLM()
    return AUTHORSHIP_PROMPT | llm | authorship_parser