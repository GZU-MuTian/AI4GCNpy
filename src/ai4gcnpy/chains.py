from . import llm_client

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union
import logging

logger = logging.getLogger(__name__)


# --- TopicLabelerChain ---

ALLOWED_PARAGRAPH_LABELS: Dict[str, str] = {
    "HeaderInformation": "Contains circular metadata.",
    "AuthorList": "Lists author names, possibly followed by affiliations or a 'on behalf of...' statement.",
    "ScientificContent": "Describes observations, analysis, results, or interpretations of an astronomical event.",
    "ExternalLinks": "Contains hyperlinks or URLs pointing to external astronomical resources",
    "ContactInformation": "Provides contact details such as email addresses or phone numbers.",
    "Acknowledgements": "Expresses gratitude for assistance or contributions (explicit or implied).",
    "CitationInstructions": "Indicates that the message is citable.",
    "Correction": "Notes about corrections or updates to previously issued information (often starts with '[GCN OP NOTE]' or 'This circular was adjusted...')."
}
# Pre-format allowed labels for use in system prompt
_allowed_paragraph_labels_str = "\n".join(
    f"- {label}: {desc}" for label, desc in ALLOWED_PARAGRAPH_LABELS.items()
)

class ParagraphLabelList(BaseModel):
    labels: List[str] = Field(description="A list of allowed labels, one per paragraph in order.")

paragraph_labels_parser = PydanticOutputParser(pydantic_object=ParagraphLabelList)

_SYSTEM_PARAGRAPH_LABEL_PROMPT = """
You are an expert astronomer analyzing NASA GCN Circulars.

**Task:** Assign exactly ONE specific topic Label to each of the numbered paragraphs provided below.

**Allowed topics (Choose Only From These):**
{allowed_labels}

**Important Instructions:**
1.  GCNs typically follow this structure:
    - 1st Paragraph: Usually `HeaderInformation` (containing TITLE, NUMBER, SUBJECT, DATE, FROM).
    - 2nd Paragraph: Usually `AuthorList`.
    - Middle Paragraph(s): Primarily `ScientificContent`.
    - Optional sections like `ExternalLinks`, `ContactInformation`, and `Acknowledgements` usually appear toward the end.
    - Final paragraphs (if present) may be 'CitationInstructions' or `Correction` information.
2.  Input Format: Each paragraph is enclosed in paired tags <PN>...</PN>, where N is the paragraph's order (1, 2, 3, ...). This numbering is for your reference to assign the correct tag based on position and content. Do NOT use any numbers found WITHIN the paragraph text to influence your decision.
3.  Output Format:
{format_instructions}
Example for 3 paragraphs: `["HeaderInformation", "AuthorList", "ScientificContent"]`
""".strip()

_HUMAN_PARAGRAPH_LABEL_PROMPT = """
**Numbered Paragraphs:**
{numbered_paragraphs}
""".strip()

PARAGRAPH_LABEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PARAGRAPH_LABEL_PROMPT),
    ("human", _HUMAN_PARAGRAPH_LABEL_PROMPT)
]).partial(
    allowed_labels=_allowed_paragraph_labels_str,
    format_instructions=paragraph_labels_parser.get_format_instructions()
)

def ParagraphLabelerChain():
    """
    Assign topic labels to paragraphs.
    """
    llm = llm_client.getLLM()
    return PARAGRAPH_LABEL_PROMPT | llm | paragraph_labels_parser

# --- ParseAuthorshipChain ---

class AuthorEntry(BaseModel):
    author: str = Field(description="Author name.")
    affiliation: str = Field(description="Institutional affiliation.")

class AuthorList(BaseModel):
    collaboration: str = Field(default="null", description="Name of the collaboration or team, or 'null' if not mentioned")
    authors: List[AuthorEntry] = Field(default_factory=list, description="List of authors and their affiliations")

author_list_parser = PydanticOutputParser(pydantic_object=AuthorList)

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
]).partial(format_instructions=author_list_parser.get_format_instructions())

def ParseAuthorshipChain():
    llm = llm_client.getLLM()
    return AUTHORSHIP_PROMPT | llm | author_list_parser

# --- ReportLabelerChain ---

# Define detailed label descriptions covering most realistic scenarios
ALLOWED_REPORT_LABELS: Dict[str, str] = {
    "NEW_EVENT_DETECTION": "The circular reports the initial detection of a new astrophysical transient or event. This includes gamma-ray bursts (GRBs), gravitational wave events (GW), neutrino, supernovae, tidal disruption events (TDEs), fast radio bursts (FRBs), or other novel cosmic phenomena. Key indicators: 'detected', 'discovered', 'triggered', 'alert', 'first report', 'new source', or similar phrasing indicating novelty and first observation.",
    "FOLLOW_UP_OBSERVATION": "The circular presents observational results (imaging, spectroscopy, photometry, timing, etc.) of a previously reported astrophysical event. This includes confirmations, multi-wavelength coverage, light curves, or spectral analysis of known transients. It may also include marginal detections or non-detections (the latter particularly when accompanied by upper limits). Key indicators: 'follow-up', 'observation of [known event]', 'counterpart candidate', 'monitoring', 'light curve', 'spectrum'.",
    "NON_DETECTION_LIMIT": "The circular explicitly states that no counterpart or signal was found for a previously reported event and provides quantitative upper limits (e.g., magnitude limits, flux limits). It may be a standalone report or part of a broader follow-up observation. Key indicators: 'no detection', 'upper limit', 'limiting magnitude', 'flux limit', 'non-detection'.",
    "ANALYSIS_REFINEMENT": "The circular refines, corrects, or improves upon earlier information about a known event. This includes updated sky localizations, revised redshifts, corrected error regions, improved classifications, or re-analyses using better calibration or methods. It m2ay also include retractions of prior event parameters if the event is real but previously reported parameters were incorrect. Key indicators: 'refined position', 'updated localization', 'revised redshift', 'corrected coordinates', 'improved analysis', 're-analysis of', 'retraction of [parameter]'.",
    "CALL_FOR_FOLLOWUP": "The circular explicitly requests or encourages other observers or facilities to conduct additional observations of a specific event. This may include time-critical requests for spectroscopy, monitoring, or multi-messenger coverage. Key indicators: 'request follow-up', 'encourage observations', 'please observe', 'urgent follow-up needed', 'call for spectroscopic coverage'.",
    "NON_EVENT_REPORT": "The circular does not pertain to any real astrophysical event. This includes system tests, software simulations, hardware injections, observatory maintenance notices, scheduling updates, mailing list announcements, false alarm retractions due to instrumental artifacts, or administrative messages. Key indicators: 'test alert', 'simulation', 'injection', 'maintenance', 'scheduling notice', 'false trigger', 'no real event', 'administrative', 'this is a test'."
}

# Pre-format allowed labels for use in system prompt
_allowed_report_labels_str = "\n".join(
    f"- {label}: {desc}" for label, desc in ALLOWED_REPORT_LABELS.items()
)

class ReportLabel(BaseModel):
    label: str = Field(..., description="The primary communication intent of the GCN Circular.")

report_label_parser = PydanticOutputParser(pydantic_object=ReportLabel)

_SYSTEM_REPORT_LABEL_PROMPT = """
You are an expert astronomer analyzing NASA GCN Circulars.
Your task is to determine the PRIMARY communication intent of the following circular.

**Allowed topics (Choose Only From These):**
{allowed_labels}

Instructions:
- Return ONLY one label that best matches the primary purpose.

{format_instructions}
""".strip()

_HUMAN_REPORT_LABEL_PROMPT = """
GCN Circular text:
{content}
""".strip()

LABEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_REPORT_LABEL_PROMPT),
    ("human", _HUMAN_REPORT_LABEL_PROMPT)
]).partial(
    allowed_labels=_allowed_report_labels_str,
    format_instructions=report_label_parser.get_format_instructions()
)

def ReportLabelerChain():
    """
    Assign topic labels to paragraphs.
    """
    llm = llm_client.getLLM()
    return LABEL_PROMPT | llm | report_label_parser

# --- ParameterExtractionChain ---

PHYSICAL_QUANTITY_CATEGORIES: Dict[str, str] = {
  "position_and_coordinates": "Source location on the sky, associated uncertainties, and angular separations. Includes coordinates (RA, Dec, J2000), error regions, and offsets.",
  "time_and_duration": "Timing of events or observations. Includes specific times (e.g., trigger T0), durations (e.g., T90), intervals, evolution indices, and timestamps (e.g., UTC, MJD).",
  "flux_and_brightness": "Observed intensity or energy reception rate. Includes flux, count rate, fluence, magnitude.",
  "spectrum_and_energy": "Photon/energy distribution vs. wavelength or energy, including spectral indices (photon/power-law index), characteristic energies (Ep, Epeak, cutoff), isotropic/rest-frame energies (Eiso), and energy bands.",
  "observation_conditions_and_instrument": "Observation setup and environment. Includes wavelength band/filter, instrumental mode, exposure time, pointing, atmospheric conditions, and instrument characteristics.",
  "distance_and_redshift": "Cosmological distance of the source. Primarily redshift, luminosity distance.",
  "extinction_and_absorption": "Light attenuation due to intervening material. Includes dust extinction (e.g., E(B-V), Av) and gas absorption (e.g., NH).",
  "statistical_significance_and_uncertainty": "Parameters quantifying the reliability, confidence, and precision of measurements. This includes: detection significance (sigma, SNR), p-values, false alarm rates, chi-squared statistics, and uncertainties with confidence levels.",
  "upper_limit": "A special class of measurement indicating that the true value of a quantity is likely below a specified threshold with a given confidence. ALWAYS used in conjunction with a physical quantity from another category. Examples: 'flux upper limit', 'magnitude upper limit', 'fluence upper limit'. Specifies the sensitivity of a non-detection.",
  "source_identification_and_characteristics": "Parameters describing the nature and environment of the astrophysical source itself. This includes: classification (GRB, GW event, neutrino candidate), association (optical/X-ray/radio counterpart), host galaxy information, and specific spectral features (emission/absorption lines like H-alpha, [OIII]) used for identification and analysis."
}

# Pre-format allowed categories for use in system prompt
_allowed_categories_str = "\n".join(
    f"- {cat}: {desc}" for cat, desc in PHYSICAL_QUANTITY_CATEGORIES.items()
)

class PhysicalQuantityCategory(BaseModel):
    position_and_coordinates: Optional[List[str]] = Field(default=None)
    time_and_duration: Optional[List[str]] = Field(default=None)
    flux_and_brightness: Optional[List[str]] = Field(default=None)
    spectrum_and_energy: Optional[List[str]] = Field(default=None)
    observation_conditions_and_instrument: Optional[List[str]] = Field(default=None)
    distance_and_redshift: Optional[List[str]] = Field(default=None)
    extinction_and_absorption: Optional[List[str]] = Field(default=None)
    statistical_significance_and_uncertainty: Optional[List[str]] = Field(default=None)
    upper_limit: Optional[List[str]] = Field(default=None)
    source_identification_and_characteristics: Optional[List[str]] = Field(default=None)


# Parser for the output
quantity_parser = PydanticOutputParser(pydantic_object=PhysicalQuantityCategory)

# System prompt for the LLM
_SYSTEM_QUANTITY_EXTRACTION_PROMPT = """
You are an expert astronomer analyzing NASA GCN Circulars.
Your task is to identify and extract specific sentences from the text that contain information about the following physical quantity categories.

**Categories to Extract:**
{allowed_categories}

**Task Instructions:**
1. Identify Sentences: Read the input text carefully and find all sentences that contain information relevant to any of the predefined categories.
2. One Category per Sentence: For each identified sentence, assign it to the ONLY ONE most specific and appropriate category. If a sentence contains information pertinent to multiple categories, identify its primary purpose and assign it only to the category that best reflects that purpose. **Do not duplicate the sentence across categories**.
3. Extract Verbatim: Copy the entire sentence exactly as it appears in the text. Do not paraphrase or modify the original wording.
4. Preserve Context: Do not truncate sentences. If the relevant information is embedded within a longer sentence, extract the full sentence. Its additional context is valuable for verification.
5. Handle Complex Sentences: If a sentence is a list joined by semicolons (;) or conjunctions (e.g., “and”, “also”), and each independent clause clearly addresses a different, distinct category, you may split it into separate entries.
6. Your output MUST be valid JSON that strictly adheres to the structure specified below.
{format_instructions}
""".strip()

# Human prompt for the LLM
_HUMAN_QUANTITY_EXTRACTION_PROMPT = """
GCN Circular text:
{content}
""".strip()

# Create the prompt template
QUANTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_QUANTITY_EXTRACTION_PROMPT),
    ("human", _HUMAN_QUANTITY_EXTRACTION_PROMPT)
]).partial(
    allowed_categories=_allowed_categories_str,
    format_instructions=quantity_parser.get_format_instructions()
)

def PhysicalQuantityExtractorChain():
    llm = llm_client.getLLM()
    return QUANTITY_EXTRACTION_PROMPT | llm | quantity_parser


# --- GuardrailsChain ---

class GuardrailsOutput(BaseModel):
    decision: Literal["gcn", "end"] = Field(
        description="Decision on whether the question is related to NASA's General Coordinates Network (GCN)"
    )

_HUMAN_CYPHER_TEMPLATE = """
The question is:
{question}
"""

_SYSTEM_GUARDRAIL_TEMPLATE = """
You are a routing agent for a specialized AI system that answers questions ONLY about NASA's General Coordinates Network (GCN).

# Domain Knowledge
The GCN distributes real-time alerts and follow-up information about:
- Gamma-ray bursts (GRBs)
- Gravitational wave events (e.g., from LIGO/Virgo)
- Neutrino alerts (e.g., from IceCube)
- Multi-messenger astrophysical transients
- Space missions like Fermi, Swift, INTEGRAL, Chandra
- GCN Notices, Circulars, and related astronomical data

# Neo4j Database Schema Context
{schema}

# Routing Logic
Determine if the user's question is about:
1. GCN domain knowledge (as described above)
2. Data stored in the Neo4j schema (nodes/relationships mentioned above)

Your task: Decide if the user's question can be answered using this GCN-focused knowledge graph.

If YES → output "gcn"
If NO → output "end"

Provide only the specified output: "gcn" or "end".
"""

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_GUARDRAIL_TEMPLATE),
    ("human", _HUMAN_CYPHER_TEMPLATE)
])

def GuardrailsChain():
    llm = llm_client.getLLM()
    return GUARDRAIL_PROMPT | llm.with_structured_output(GuardrailsOutput)

# --- CypherChain ---

_SYSTEM_CYPHER_TEMPLATE = """
You are a Neo4j expert.
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Use only the provided relationship types and properties in the schema:
{schema}

Key Facts:
- All factual content from a Circular has been categorized into `HAS_PHYSICAL_QUANTITY` or `HAS_METADATA` relationships. Do NOT read `CIRCULAR.rawText` — all relevant information is already organized into the above relationships.
- The `sentences` property on these relationships contains the **exact original sentence(s)** from the Circular that support the category.
- To answer a question, identify the most relevant semantic category (e.g., "observation_conditions_and_instrument" for instrument questions), then retrieve `r.sentences`.
- If the question references a specific circular (e.g., "GCN 12345"), match by `circularId`.
- If the question is about a specific GRB or astrophysical event (e.g., "GRB 221009A"), locate relevant CIRCULAR nodes by checking if the event name appears in the `subject` property.
  - Use case-insensitive substring matching: `toLower(c.subject) CONTAINS toLower($eventName)`
  - Avoid full-text scan of `rawText`; `subject` is sufficient for event identification.

Guidelines:
- Question: "What instruments were used?" → use category = 'observation_conditions_and_instrument'
- Question: "What is the position?" → use 'position_and_coordinates'
- Question: "Who to contact?" → use METADATA with type = 'contactInformation'
- Once the relevant CIRCULAR(s) are found, query their `HAS_INTENT` `HAS_PHYSICAL_QUANTITY` or `HAS_METADATA` relationships to retrieve answers. Return circularId, subject, createdOn, intent, relationship type, unified category, and sentences.

Rules:
- Do not include explanations, comments, markdown, backticks, or any text beyond the Cypher statement.
- NEVER return entire nodes unless the question explicitly asks for a full description.
- When querying multiple CIRCULAR nodes, always sort results by `c.createdOn` in descending order (newest first)  unless the question implies otherwise.
"""

CYPHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_CYPHER_TEMPLATE),
    ("human", _HUMAN_CYPHER_TEMPLATE)
])


def Text2CypherChain():
    llm = llm_client.getLLM()
    return CYPHER_PROMPT | llm | StrOutputParser()

# --- ValidateCypherChain ---

_SYSTEM_VALIDATE_CYPHER_TEMPLATE = """
You are a Cypher expert reviewing a statement written by a junior developer.
"""

_HUMAN_VALIDATE_CYPHER_TEMPLATE = """
You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relationship types missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{schema}

The question is:
{question}

The Cypher statement is:
{cypher}

Make sure you don't make any mistakes!
"""

VALIDATE_CYPHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_VALIDATE_CYPHER_TEMPLATE),
    ("human", _HUMAN_VALIDATE_CYPHER_TEMPLATE)
])

def ValidateCypherChain():
    llm = llm_client.getLLM()
    return VALIDATE_CYPHER_PROMPT | llm | StrOutputParser()

# --- CorrectCypherChain ---

_SYSTEM_CORRECT_CYPHER_PROMPT = """
You are a Cypher expert reviewing a statement written by a junior developer. You need to correct the Cypher statement based on the provided errors. No pre-amble. Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
"""

_HUMAN_CORRECT_CYPHER_PROMPT = """
Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: 
"""

CORRECT_CYPHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_CORRECT_CYPHER_PROMPT),
    ("human", _HUMAN_CORRECT_CYPHER_PROMPT)
])

def CorrectCypherChain():
    llm = llm_client.getLLM()
    return CORRECT_CYPHER_PROMPT | llm | StrOutputParser()

# --- GenerateFinalChain ---

_SYSTEM_GENERATE_FINAL_PROMPT = """
You are a helpful assistant specialized in NASA's General Coordinates Network (GCN).

Answer the user's question based ONLY on the provided context. If the context is empty or irrelevant, say you cannot answer.
"""

_HUMAN_GENERATE_FINAL_PROMPT = """
Question: 
{question}

Results: 
{results}
"""

GENERATE_FINAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_GENERATE_FINAL_PROMPT),
    ("human", _HUMAN_GENERATE_FINAL_PROMPT)
])

def GenerateFinalChain(): 
    llm = llm_client.getLLM()
    return GENERATE_FINAL_PROMPT | llm | StrOutputParser()