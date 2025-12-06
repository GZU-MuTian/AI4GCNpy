from ai4gcnpy import gcn_extractor

from rich.progress import track
from pathlib import Path
import json

dir_path = Path("/home/yuliu/Downloads/archive.txt")
txt_files = list(dir_path.glob("*.txt"))
# selected_file = random.choice(txt_files)

processed_count: int = 0
error_count: int = 0

output_dir = Path("/home/yuliu/Downloads/gcn_results")
output_dir.mkdir(parents=True, exist_ok=True)
error_log = output_dir / "gcn_errors.log"
for idx, file in enumerate(track(txt_files, description="Processing...", transient=True), start=1):
    # if idx % 20 == 0:
    #     continue
    try:
        result = gcn_extractor(file, model="deepseek-r1:8b", model_provider="ollama")

        # Write result as JSON
        output_file = output_dir / f"{file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        processed_count += 1
    except Exception as e:
        error_count += 1
        error_msg: str = f"Failed to process {file}: {str(e)}"
        with open(error_log, "a", encoding="utf-8") as log_f:
            log_f.write(f"{error_msg}\n")

print(f"Processing complete: {processed_count} succeeded, {error_count} failed. ")