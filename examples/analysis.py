import json
import logging
from pathlib import Path
from typing import Any, Set
from pydantic import BaseModel, Field, validator

# Basic logger setup (minimal color via default stream)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

dir = Path("/home/yuliu/Downloads/gcn_results")
files = list(dir.glob("*.json"))
refs = []
contacts =[]
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Skip invalid file {f}: {e}")
        continue
    if data.get("extracted_dset"):
        for k, vl in data["extracted_dset"].items():
            if k == "references":
                for v in vl:
                    if v["type"] not in ["image", "data", "analysis", "catalog", "lightcurve", "spectrum"]:
                        print()
                        # print(data.get("paragraphs")["References"], "\n\n")
                    # refs.append()
            elif k == "contacts":
                for v in vl:
                    print(data.get("paragraphs")["ContactInformation"], "\n\n")
                    contacts.append(v)
    else:
        print("Error file", f)

# ref_types = set()
# for ref in refs:
#     ref_types.add(ref["type"])
contact_types = set()
for contact in contacts:
    contact_types.add(contact["type"])
print(contact_types)
# print(refs)
