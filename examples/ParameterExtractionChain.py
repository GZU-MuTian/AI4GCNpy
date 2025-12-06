from ai4gcnpy import llm_client

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List, Set, Optional, Dict
from enum import Enum
from rich.progress import track
from pathlib import Path
from rich.logging import RichHandler
import logging


# Configure logging
log_file = Path("~/Desktop/gcn_parameter_extraction.log").expanduser()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
# 🔇 Suppress noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# === LLM Setup ===
llm_config = {
    "model": "deepseek-r1:8b",
    "model_provider": "ollama",
}
llm_client.basicConfig(**llm_config)
llm = llm_client.getLLM()


class ParameterExtraction(BaseModel):
    parameter: str = Field(..., description="The physical quantity mentioned (e.g., 'redshift', 'X-ray flux', 'magnitude'). Use the term as close to the original text as possible.")
    supporting_text: str = Field(..., description="The exact sentence where this parameter appears.")

class ExtractionList(BaseModel):
    extractions: List[ParameterExtraction] = Field(default_factory=list, description="List of all extracted parameters")

# Use parser for list output
output_parser = PydanticOutputParser(pydantic_object=ExtractionList)

prompt_template = """
You are an expert in astrophysics and Gamma-ray Burst (GRB) follow-up observations.
Your task is to identify ALL physical quantities or observables mentioned in the following NASA GCN Circular.

Examples of physical quantities:
- redshift (z)
- X-ray flux, X-ray upper limit
- optical magnitude (e.g., r=20.5)
- spectral index
- polarization

Instructions:
1. Extract EVERY physical quantity that appears, even if only implied or used in an upper limit.
2. For each, provide:
   - `parameter`: The name of the quantity (keep it simple and close to how it's referred to in astronomy).
   - `supporting_text`: The FULL original sentence.
3. Do NOT invent quantities not present.
4. Return a list, even if empty.

GCN Circular:
{circular_content}

{format_instructions}
"""
prompt = PromptTemplate(
    input_variables=["circular_content"],
    template=prompt_template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)
chain = prompt | llm | output_parser


# Randomly select one file
dir_path = Path("/home/yuliu/Downloads/archive.txt")
txt_files = list(dir_path.glob("*.txt"))


params: Set[str] = set()

for idx, file in enumerate(track(txt_files, description="Processing...", transient=True), start=1):
    content = Path(file).read_text(encoding="utf-8")

    try:
        response: ExtractionList = chain.invoke({"circular_content": content})
        
        for item in response.extractions:
            if item.parameter not in params:
                logger.info(f"{item.parameter} | Supporting text: {item.supporting_text}")
                params.add(item.parameter)

    except Exception as e:
        logger.error(f"Failed to parse LLM response for {file.name}")
        continue

