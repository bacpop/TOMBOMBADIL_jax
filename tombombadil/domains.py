"""Minimal alignment-to-domain mapping for per-site omega plot colouring.

This deliberately contains annotation and plotting labels only; it does not
define or fit a domain regression model.
"""

import json
import warnings

import numpy as np


def _read_fasta(path):
    sequence = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                sequence.append(line)
    return "".join(sequence)


def _read_alignment(path):
    records = []
    header = None
    sequence = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence)))
                header, sequence = line[1:], []
            elif line:
                sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def parse_domain_labels(json_path, alignment_path, reference_path, n_sites):
    """Return ``other``/``extracellular`` labels for alignment codon sites."""
    records = _read_alignment(alignment_path)
    reference_length = len(_read_fasta(reference_path))
    if not records or reference_length <= 0:
        raise ValueError("Alignment and non-empty reference protein are required")

    proxy = min(records, key=lambda item: abs(
        sum(proxy_codon != "---" for proxy_codon in
            (item[1][i:i + 3] for i in range(0, len(item[1]) - 2, 3)))
        - reference_length
    ))[1]
    alignment_sites = len(proxy) // 3
    labels = np.full(alignment_sites, "other", dtype=object)
    protein_position = 0
    column_to_position = np.full(alignment_sites, -1, dtype=int)
    for column in range(alignment_sites):
        if proxy[column * 3:(column + 1) * 3] != "---":
            protein_position += 1
            column_to_position[column] = protein_position

    with open(json_path) as handle:
        data = json.load(handle)
    extracellular = set()
    for feature in data.get("features", []):
        if feature.get("description", "").lower() != "extracellular":
            continue
        location = feature.get("location", {})
        start = location.get("start", {})
        end = location.get("end", {})
        if "value" not in start or "value" not in end:
            warnings.warn("Skipping extracellular feature without exact bounds")
            continue
        extracellular.update(range(int(start["value"]), int(end["value"]) + 1))

    for column, position in enumerate(column_to_position):
        if position in extracellular:
            labels[column] = "extracellular"
    if len(labels) != n_sites:
        raise ValueError(f"Domain labels have length {len(labels)}, expected {n_sites}")
    return labels
