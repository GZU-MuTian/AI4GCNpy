from . import llm_client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
import logging

logger = logging.getLogger(__name__)


# --- TopicLabelerChain ---

_ALLOWED_LABEL = Literal[
    "HeaderInformation",
    "AuthorList",
    "ScientificContent",
    "References",
    "ContactInformation",
    "Acknowledgements",
    "CitationInstructions",
    "Correction"
]
class LabelList(BaseModel):
    labels: List[_ALLOWED_LABEL] = Field(description="A list of allowed labels, one per paragraph in order.")

labels_parser = PydanticOutputParser(pydantic_object=LabelList)

_SYSTEM_LABEL_PROMPT = """
You are an expert astronomer analyzing NASA GCN Circulars.

**Task:** Assign exactly ONE specific topic Label to each of the numbered paragraphs provided below.

**Allowed topics (Choose Only From These):**
- HeaderInformation: Contains circular metadata.
- AuthorList: Lists author names, possibly followed by affiliations or a "on behalf of..." statement.
- ScientificContent: Describes observations, analysis, results, or interpretations of an astronomical event.
- References: Contains links to external astronomical resources.
- ContactInformation: Provides contact details such as email addresses or phone numbers.
- Acknowledgements: Expresses gratitude for assistance or contributions.
- CitationInstructions: Indicates that the message is citable.
- Correction: Notes about corrections or updates to previously issued information (often starts with "[GCN OP NOTE]" or "This circular was adjusted...").

**Important Instructions:**
1.  GCNs typically follow this structure:
    - 1st Paragraph: Usually `HeaderInformation` (containing TITLE, NUMBER, SUBJECT, DATE, FROM).
    - 2nd Paragraph: Usually `AuthorList`.
    - Middle Paragraph(s): Primarily `ScientificContent`.
    - Optional sections like `References`, `ContactInformation`, and `Acknowledgements` usually appear toward the end.
    - Final paragraphs (if present) may be 'CitationInstructions' or `Correction` information.
2.  Input Format: Each paragraph is enclosed in paired tags <PN>...</PN>, where N is the paragraph's order (1, 2, 3, ...). This numbering is for your reference to assign the correct tag based on position and content. Do NOT use any numbers found WITHIN the paragraph text to influence your decision.
3.  Output Format:
{format_instructions}
Example for 3 paragraphs: `["HeaderInformation", "AuthorList", "ScientificContent"]`
""".strip()


_HUMAN_LABEL_PROMPT = """
**Numbered Paragraphs:**
{numbered_paragraphs}
""".strip()

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

# --- ParseAuthorshipChain ---

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
""".strip()

_HUMAN_AUTHORSHIP_PROMPT = """
**Input Text:**
{content}
""".strip()

AUTHORSHIP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_AUTHORSHIP_PROMPT),
    ("human", _HUMAN_AUTHORSHIP_PROMPT)
]).partial(format_instructions=authorship_parser.get_format_instructions())

def ParseAuthorshipChain():
    llm = llm_client.getLLM()
    return AUTHORSHIP_PROMPT | llm | authorship_parser

# --- ParseReferenceChain ---

class Reference(BaseModel):
    "Represents a single reference URL extracted from a GCN Circular."
    type: Union[Literal["image", "data", "report", "catalog", "lightcurve", "spectrum"], str] = Field(description="Type of information.")
    url: str = Field(description="The exact URL as it appears in the original text.")

class ReferenceList(BaseModel):
    references: List[Reference] = Field(default_factory=list)

reference_parser = PydanticOutputParser(pydantic_object=ReferenceList)

_SYSTEM_REFERENCE_PROMPT = """
You are an expert astronomer analyzing a NASA GCN Circular.
Extract **all** URLs that appear **verbatim** in the provided text.
Do NOT generate, complete, infer, or paraphrase URLs.

Focus on links mentioned after phrases like:
- "available at"
- "found at"
- "posted at"
- "can be obtained at"

If no URLs are found, return an empty array: [].

{format_instructions}
""".strip()

_HUMAN_REFERENCE_PROMPT = """
Extract any reference URL from the following GCN Circular excerpt:

{content}
""".strip()

REFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_REFERENCE_PROMPT),
    ("human", _HUMAN_REFERENCE_PROMPT)
]).partial(format_instructions=reference_parser.get_format_instructions())

def ParseReferenceChain():
    llm = llm_client.getLLM()
    logger.debug("Building reference parsing chain...")
    return REFERENCE_PROMPT | llm | reference_parser

# --- ParseContactINFOChain ---

class ContactItem(BaseModel):
    """
    A single contact entry: either an email or a phone number.
    """
    type: Union[Literal["email", "phone"], str] = Field(description="Type of contact")
    value: str = Field(description="Exact string from the original text")

class ContactList(BaseModel):
    contacts: List[ContactItem] = Field(default_factory=list)

contact_info_parser = PydanticOutputParser(pydantic_object=ContactList)

_SYSTEM_CONTACTINFO_PROMPT = """
You are an expert astronomer analyzing a NASA GCN Circular.
Extract **all** email addresses and phone numbers that appear **exactly** in the provided text.
Do NOT guess, correct formatting, or invent contact details. Only include what is literally present.

Look near phrases like:
- "contact"
- "direct communications to"
- "for further information"

If no contacts are found, return an empty array: [].

{format_instructions}
"""

_HUMAN_CONTACTINFO_PROMPT = """
Extract contact information from the following GCN Circular excerpt:

{content}
""".strip()

CONTACTINFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_CONTACTINFO_PROMPT),
    ("human", _HUMAN_CONTACTINFO_PROMPT)
]).partial(format_instructions=contact_info_parser.get_format_instructions())

def ParseContactINFOChain():
    llm = llm_client.getLLM()
    logger.debug("Building contact information parsing chain...")
    return CONTACTINFO_PROMPT | llm | contact_info_parser
