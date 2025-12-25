# AI4GCNpy: A Graph Database Toolkit for NASA's GCN

[![PyPI version](https://badge.fury.io/py/ai4gcnpy.svg)](https://pypi.org/project/ai4gcnpy/)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AI4GCNpy is a Python toolkit for building and querying a knowledge graph of astrophysical transient events from NASA's Gamma-ray Coordinates Network (GCN). Powered by LangGraph and Neo4j technology, it enables natural language querying for astrophysical transient events.

**Key capabilities:**
- **Automatic Information Extraction**: Extract structured information from GCN circulars using LLMs.
- **Knowledge Graph Construction**: Convert unstructured GCN circulars into a structured Neo4j graph database.
- **Intelligent Q&A System**: Converts natural language questions into Cypher via LLM, executes graph queries in Neo4j, and generates final answers by combining structured results with relevant passages from the original GCN circulars.
- **Beautiful Output**: Colorful terminal output, syntax highlighting, and progress bars using the Rich library.

## Quick Start

### Install

Install the package from PyPI:
```bash
pip install ai4gcnpy
```

Or install from source for development:
```bash
git clone https://github.com/GZU-MuTian/AI4GCNpy.git
cd AI4GCNpy
pip install -e .
```

### Set Up Neo4j (Required)

AI4GCNpy requires a locally running Neo4j instance as its graph database backend. Note that the APOC (Awesome Procedures On Cypher) plugin is required for advanced graph operations.

### Environment Setup (Recommended)

To streamline usage and avoid repetitive CLI flags, we recommend configuring environment variables. This approach simplifies command execution and enhances security by avoiding credentials in command history.

```bash
# LLM Configuration (required)
DEEPSEEK_API_KEY="your-deepseek-api-key-here"

# Neo4j Configuration (required for graph operations)
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-neo4j-password"

# Optional: Set default output directory
GCN_OUTPUT_PATH="./gcn_data"
```

> You may also pass these values directly via CLI flags (e.g., --url, --username, --password).

## Usage Guide

### Core Functions

The `ai4gcnpy` package provides three core functions for processing of NASA GCN data:
```python
from ai4gcnpy import gcn_extractor, gcn_builder, gcn_graphrag
```
> These functions form a complete pipeline:
> Extract → Build → Query, enabling structured knowledge extraction, graph population, and natural-language question answering.

1. Extract structured data from GCN circulars:
```python
from ai4gcnpy import gcn_extractor
import json

# Extract information from a single GCN circular
result = gcn_extractor(
    input_file="path/to/gcn_circular.txt",
    model="deepseek-chat",
    model_provider="deepseek",
    temperature=0.7,
    max_tokens=4000,
    reasoning=True
)

# Save the extracted data
if result:
    output_file = "path/to/extracted_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
```
> The `model` and `model_provider` parameters are passed to LangChain’s unified chat model initializer: `langchain.chat_models.init_chat_model`. This allows `ai4gcnpy` to support multiple LLM providers (e.g., DeepSeek, OpenAI, Anthropic) through a consistent interface, while abstracting provider-specific setup details.

2. Populate Neo4j with extracted data:
```python
from ai4gcnpy import gcn_builder

gcn_builder(
    json_file="path/to/extracted_data.json",
    database="neo4j"  # Optional: specify database (defaults to 'neo4j')
)
```

3. Ask questions using natural language:
```python
from ai4gcnpy import gcn_graphrag

response = gcn_graphrag(
    query_text="Your question here",
    model="deepseek-chat",
    model_provider="deepseek"
    database="neo4j"
)

print(f"Query: {response.get('query')}")
print(f"Cypher: {response.get('cypher_statement')}")
print(f"Data Sources: {response.get('retrieved_chunks')}")
print(f"Final Answer: {response.get('answer')}")
```

### Command-Line Interface

For rapid prototyping or batch workflows, `ai4gcnpy` includes a CLI named `gcn-cli`. It uses the same core functions as the Python API—ensuring consistent behavior across interfaces.

> 🔧 Tip: Run `gcn-cli --help` for an overview, or `gcn-cli <command> --help` for command-specific options.

Basic Commands:
```bash
# Extract information from GCN circulars
gcn-cli extractor path/to/gcn_circular.txt

# Batch extract from multiple files
gcn-cli batch_extractor --input path/to/circulars_directory/ --output path/to/extracted_data_directory/

# Build graph
gcn-cli builder path/to/extracted_data_directory/

# Ask a question
gcn-cli query "Your question here"
```

Adjust verbosity for debugging or quiet runs:
```bash
# Production - errors only
gcn-cli --log-level ERROR query "Your question here"

# Short form for debugging
gcn-cli -v DEBUG query "Your question here"
```

## Project Structure

```
ai4gcnpy/
├── agents.py        # LangGraph agents for complex workflows
├── chains.py        # LangChain chains for LLM interactions
├── cli.py           # Command-line interface built with Typer
├── core.py          # Core functions
├── db_client.py     # Neo4j database connector
├── llm_client.py    # Unified LLM provider interface
├── utils.py         # Utility functions (e.g., download_gcn_archive)
```

## Related Resources

- NASA GCN Archive: https://gcn.nasa.gov/circulars/
- Neo4j Documentation: https://neo4j.com/docs/
- Neo4j Linux installation: https://neo4j.com/docs/operations-manual/current/installation/linux/debian/
- LangGraph Guide: https://docs.langchain.com/

## Contact

For questions and support:

- Author: Yu Liu
- Email: yuliu@gzu.edu.cn
- Repository: https://github.com/GZU-MuTian/AI4GCNpy

Astronomical Discovery, Powered by AI! 🔭✨

