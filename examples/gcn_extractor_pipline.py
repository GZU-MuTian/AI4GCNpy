import json
import logging
from pathlib import Path
from rich.progress import track

# Import your custom extractor (assumed to be available)
try:
    from ai4gcnpy import gcn_extractor
except ImportError as e:
    raise ImportError(
        "Required package 'ai4gcnpy' not found. Please install it."
    ) from e


# Ensure output directory exists
output_dir = Path("/home/yuliu/Downloads/gcn_results")
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / "gcn_extractor.log"

logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 写入文件
    ]
)
logger = logging.getLogger(__name__)

# Discover input files
input_dir = Path("/home/yuliu/Downloads/archive.txt")
txt_files = list(input_dir.glob("*.txt"))

for file in track(txt_files, description="Processing files...", transient=True):
    try:
        result = gcn_extractor(
            str(file),
            model="qwen3:8b",
            model_provider="ollama",
            reasoning=False
        )
        # Write result as JSON
        output_file = output_dir / f"{file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to process {file}: {str(e)}")
        continue