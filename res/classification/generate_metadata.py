"""Generate metadata.csv from all .arff and .arff.gz files in res/classification subfolders."""

import re
import csv
import gzip
from pathlib import Path

BASE_DIR = Path(__file__).parent


def open_arff(filepath):
    """Open an ARFF as text, gzipped or not."""
    if str(filepath).endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")
    return open(filepath, "r", encoding="utf-8", errors="replace")


def parse_arff(filepath):
    """Extract metadata from an .arff file by parsing its header and counting data rows."""
    attributes = []
    in_data = False
    n_instances = 0

    with open_arff(filepath) as f:
        for line in f:
            stripped = line.strip()

            if not stripped or stripped.startswith("%"):
                if in_data and stripped.startswith("%"):
                    continue
                continue

            upper = stripped.upper()

            if upper.startswith("@ATTRIBUTE"):
                match = re.match(
                    r"@attribute\s+(\S+)\s+(.*)", stripped, re.IGNORECASE
                )
                if match:
                    attr_type = match.group(2).strip()
                    if attr_type.startswith("{"):
                        attributes.append("nominal")
                    elif attr_type.upper() in ("REAL", "NUMERIC", "INTEGER", "CONTINUOUS"):
                        attributes.append("numeric")
                    elif attr_type.upper() == "STRING":
                        attributes.append("string")
                    elif attr_type.upper().startswith("DATE"):
                        attributes.append("date")
                    else:
                        attributes.append("numeric")
            elif upper.startswith("@DATA"):
                in_data = True
            elif in_data:
                n_instances += 1

    n_attributes = len(attributes)
    disc_atts = sum(1 for a in attributes if a == "nominal")
    num_atts = sum(1 for a in attributes if a in ("numeric", "date", "string"))

    # nClasses = number of values of the last (class) attribute
    n_classes = 0
    if attributes:
        # Re-read to get the class attribute values
        with open_arff(filepath) as f:
            last_attr_line = None
            for line in f:
                stripped = line.strip()
                if re.match(r"@attribute\s+", stripped, re.IGNORECASE):
                    last_attr_line = stripped
                if stripped.upper().startswith("@DATA"):
                    break

            if last_attr_line:
                match = re.match(
                    r"@attribute\s+\S+\s+(.*)", last_attr_line, re.IGNORECASE
                )
                if match:
                    attr_type = match.group(1).strip()
                    if attr_type.startswith("{"):
                        values = attr_type.strip("{}").split(",")
                        n_classes = len([v.strip() for v in values if v.strip()])
                    else:
                        # Numeric class — count unique values from data
                        n_classes = count_unique_class_values(filepath)

    return n_instances, n_attributes, n_classes, disc_atts, num_atts


def count_unique_class_values(filepath):
    """Count unique values of the last column in @DATA section."""
    unique = set()
    in_data = False
    with open_arff(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper().startswith("@DATA"):
                in_data = True
                continue
            if in_data and stripped and not stripped.startswith("%"):
                # Last value before newline
                parts = stripped.rstrip().split(",")
                if parts:
                    unique.add(parts[-1].strip())
    return len(unique)


def main():
    rows = []
    paths = list(BASE_DIR.rglob("*.arff")) + list(BASE_DIR.rglob("*.arff.gz"))
    for arff_path in sorted(paths, key=lambda p: p.with_name(p.name.removesuffix(".gz"))):
        subfolder = arff_path.parent.relative_to(BASE_DIR).as_posix()
        print(f"Processing {subfolder}/{arff_path.name}...")
        n_instances, n_attributes, n_classes, disc_atts, num_atts = parse_arff(arff_path)
        rows.append({
            "name": arff_path.name.removesuffix(".gz"),
            "subfolder": subfolder,
            "nInstances": n_instances,
            "nAttributes": n_attributes,
            "nClasses": n_classes,
            "discAtts": disc_atts,
            "numAtts": num_atts,
        })

    output_path = BASE_DIR / "metadata.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "subfolder", "nInstances", "nAttributes", "nClasses", "discAtts", "numAtts"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} entries to {output_path}")


if __name__ == "__main__":
    main()
