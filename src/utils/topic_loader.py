"""
Topic-loading interface: give the pipeline a plain list of topics instead
of wiring up a keyword-expansion system. One topic per line in a .txt file,
or a 'topic' column in a .csv — pick whichever is more convenient.
"""
import csv
from pathlib import Path


def load_topics(source_file: str, limit: int = 10) -> list[str]:
    path = Path(source_file)
    if not path.exists():
        print(f"Warning: topics file {path} not found.")
        return []

    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "topic" in reader.fieldnames:
                topics = [row["topic"].strip() for row in reader if row.get("topic", "").strip()]
            else:
                f.seek(0)
                topics = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    else:
        with open(path, encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    return topics[:limit]
