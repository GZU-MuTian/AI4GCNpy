import re
from typing import List, Optional


pattern = re.compile(r"^([^|]+)\s*\|\s*Supporting text:")


parameters: List[str] = []
log_file_path = "/home/yuliu/Desktop/gcn_parameter_extraction.log"
with open(log_file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            param_value = match.group(1).strip()
            parameters.append(param_value)
        
print(parameters)