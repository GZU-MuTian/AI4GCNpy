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


log_file = Path("~/Desktop/gcn_labeling.log").expanduser()
# Configure logging
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

llm_config = {
    "model": "deepseek-r1:8b",
    "model_provider": "ollama",
}
llm_client.basicConfig(**llm_config)
llm = llm_client.getLLM()



class LabelManager:
    """Manages a dynamic set of purpose/intent labels with descriptive text."""

    def __init__(self, initial_labels: Dict[str, str]):
        self.existing_labels: Dict[str, str] = initial_labels

    def has_label(self, name: str) -> bool:
        """Check if a label name already exists."""
        return name.strip() in self.existing_labels

    def add_label(self, name: str, description: Optional[str] = None) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Label name cannot be empty.")
        if description is None:
            description = name  # fallback or use "" if preferred
        self.existing_labels[name] = description.strip()

    def get_labels_as_str(self) -> str:
        lines = []
        for name in sorted(self.existing_labels):
            desc = self.existing_labels[name]
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
    

class LabelResponse(BaseModel):
    name: str = Field(..., description="A short, canonical name for the intent label (e.g., 'NEW_DISCOVERY'). Use uppercase with underscores.")
    description: str = Field(..., description="A clear, concise English sentence describing the purpose or intent of this label.")
output_parser = PydanticOutputParser(pydantic_object=LabelResponse)

prompt_template = """
Analyze the following NASA GCN Circular and classify its primary communication intent or purpose.

Available labels (use one of these if it matches well):
{initial_labels}

Instructions:
1. If the circular clearly matches one of the available labels, reuse that label's name and description exactly.
2. **Only create a new label if the circular's intent is fundamentally different from ALL available labels.**
   - The 'name' must be a short, machine-readable identifier in UPPER_SNAKE_CASE (e.g., 'NEW_DISCOVERY').
   - The 'description' must be a single, clear English sentence explaining the intent.
3. Output ONLY a valid JSON object with 'name' and 'description' fields. No other text.

GCN Circular:
{circular_content}

{format_instructions}
"""
prompt = PromptTemplate(
    input_variables=["circular_content", "initial_labels"],
    template=prompt_template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)
chain = prompt | llm | output_parser


# Randomly select one file
dir_path = Path("/home/yuliu/Downloads/archive.txt")
txt_files = list(dir_path.glob("*.txt"))

initial_label_dict = {
    "NEW_DISCOVERY": "New astronomical discovery",
    "FOLLOW_UP_PROPOSAL": "Follow-up observation proposal",
    "FUTURE_PLANS": "Sharing future observation plans"
}
label_manager = LabelManager(initial_labels=initial_label_dict)

for idx, file in enumerate(track(txt_files, description="Processing...", transient=True), start=1):
    content = Path(file).read_text(encoding="utf-8")

    labels_str = label_manager.get_labels_as_str()
    try:
        response: LabelResponse = chain.invoke({
            "circular_content": content,
            "initial_labels": labels_str
        })
    except Exception as e:
        logger.error(f"Failed to parse LLM response for {file.name}")
        continue

    # if predicted_label in label_manager.existing_labels:
    if label_manager.has_label(response.name):
        continue
    else:
        # Treat the entire string as both name and description (or parse if needed)
        label_manager.add_label(response.name, response.description)
        logger.info(f"Added new label from {file.name}: {response.name} → {response.description}")

