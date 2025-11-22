from ai4gcnpy.agents import GCNParserGraph, GCNState

from langchain_ollama import ChatOllama

from pathlib import Path
import random
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Randomly select one file
dir_path = Path("/home/yuliu/Downloads/archive.txt")
txt_files = list(dir_path.glob("*.txt"))
selected_file = random.choice(txt_files)
logging.info("Randomly selected .txt file: %s", selected_file)


doc = Path(selected_file).read_text(encoding="utf-8")

llm = ChatOllama(
    model="qwen3:8b",
    temperature=1,
    reasoning=True,
    validate_model_on_init=True
)

initial_input = GCNState(
    raw_text=doc,
)

app = GCNParserGraph(llm=llm).compile()


final_state_dict = app.invoke(input=initial_input.model_dump())

grouped_paragraphs = final_state_dict.get("grouped_paragraphs", "")
for tag, merged_text in grouped_paragraphs.items():
    print(f"\n--- Tags: {tag} ---")
    print(merged_text)