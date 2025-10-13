import pandas as pd
from pathlib import Path
from utils.logger import setup_logging

logger = setup_logging()

def parse_index_file(index_path: Path):
    records = []
    with open(index_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) < 5:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pdb_id = parts[0].lower()
                resolution = float(parts[1])
                year = int(parts[2])
                affinity = float(parts[3])
                records.append({
                    "complex_id": pdb_id,
                    "resolution": resolution,
                    "year": year,
                    "affinity": affinity,
                    "measure": "pKd",
                })
            except Exception as e:
                logger.warning(f"Skipping malformed line: {line.strip()} ({e})")

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} entries from {index_path.name}")
    return df

