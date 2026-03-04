import json
import warnings
import numpy as np


def _read_protein_fasta(fasta_path):
    """Read a single protein sequence from a FASTA file. Returns (header, sequence)."""
    header, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            elif line:
                seq.append(line)
    return header, "".join(seq)


def _read_dna_alignment(alignment_path):
    """Read all sequences from a DNA FASTA alignment. Returns list of (header, sequence)."""
    sequences = []
    header, seq = None, []
    with open(alignment_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq)))
                header, seq = line[1:], []
            elif line:
                seq.append(line)
    if header is not None:
        sequences.append((header, "".join(seq)))
    return sequences


def _non_gap_codon_count(dna_seq):
    """Count codons in a DNA sequence that are not all-gap (---)."""
    n_codons = len(dna_seq) // 3
    return sum(1 for i in range(n_codons) if dna_seq[i*3:(i+1)*3] != "---")


def _gap_codon_mask(dna_seq):
    """Return bool array of length n_codons: True where codon is all-gap (---)."""
    n_codons = len(dna_seq) // 3
    return np.array([dna_seq[i*3:(i+1)*3] == "---" for i in range(n_codons)])


def build_alignment_to_protein_map(alignment_path, reference_protein_path):
    """Build a mapping from alignment column (0-based codon index) to protein position (1-based).

    Finds the first sequence in the alignment whose number of non-gap codons matches
    the reference protein length, and uses its gap pattern to determine which alignment
    columns correspond to real protein positions vs. insertions relative to the reference.

    Args:
        alignment_path: path to codon DNA alignment (FASTA format, gaps as ---)
        reference_protein_path: path to reference protein sequence (FASTA format)

    Returns:
        col_to_protein_pos: int32 array of shape (n_sites,) where entry i is the 1-based
                            protein position for alignment column i, or -1 if the column
                            is an insertion relative to the reference
    """
    _, ref_protein = _read_protein_fasta(reference_protein_path)
    ref_len = len(ref_protein)

    sequences = _read_dna_alignment(alignment_path)
    if not sequences:
        raise ValueError(f"No sequences found in {alignment_path}")

    n_sites = len(sequences[0][1]) // 3

    # Find first sequence whose non-gap codon count matches the reference protein length
    proxy_seq = None
    for header, dna_seq in sequences:
        if _non_gap_codon_count(dna_seq) == ref_len:
            proxy_seq = dna_seq
            break

    if proxy_seq is None:
        counts = [_non_gap_codon_count(s) for _, s in sequences]
        best_header, best_seq = min(sequences, key=lambda x: abs(_non_gap_codon_count(x[1]) - ref_len))
        best_count = _non_gap_codon_count(best_seq)
        warnings.warn(
            f"No alignment sequence has exactly {ref_len} non-gap codons (reference protein length). "
            f"Using the closest match ({best_count} non-gap codons). "
            f"Domain mapping may be inaccurate — check alignment and reference."
        )
        proxy_seq = best_seq

    # Build mapping using gap pattern of the proxy sequence
    is_gap = _gap_codon_mask(proxy_seq)
    col_to_protein_pos = np.full(n_sites, -1, dtype=np.int32)
    protein_pos = 0
    for i in range(n_sites):
        if not is_gap[i]:
            protein_pos += 1
            col_to_protein_pos[i] = protein_pos  # 1-based

    return col_to_protein_pos


def parse_domain_json(json_path, alignment_path, reference_protein_path, n_sites):
    """Parse a UniProt JSON file and return a binary array indicating extracellular sites.

    Uses a reference protein sequence and the codon alignment to correctly map UniProt
    protein positions (1-based) to alignment column indices, accounting for insertions
    relative to the reference.

    Args:
        json_path: path to UniProt JSON file
        alignment_path: path to codon DNA alignment (FASTA format)
        reference_protein_path: path to reference protein sequence (FASTA format)
        n_sites: number of alignment sites (codon positions), used for validation

    Returns:
        is_extracellular: float64 numpy array of shape (n_sites,), 1=extracellular, 0=other
    """
    col_to_protein_pos = build_alignment_to_protein_map(alignment_path, reference_protein_path)

    if len(col_to_protein_pos) != n_sites:
        raise ValueError(
            f"Alignment has {len(col_to_protein_pos)} codon columns but n_sites={n_sites}"
        )

    with open(json_path) as f:
        data = json.load(f)

    # Collect all extracellular protein positions (1-based)
    extracellular_positions = set()
    for feature in data.get("features", []):
        desc = feature.get("description", "")
        if desc.lower() != "extracellular":
            continue

        loc = feature.get("location", {})
        start = loc.get("start", {})
        end = loc.get("end", {})

        if start.get("modifier") != "EXACT" or end.get("modifier") != "EXACT":
            warnings.warn(
                f"Skipping non-EXACT feature at positions "
                f"{start.get('value')}-{end.get('value')}: {desc!r}"
            )
            continue

        extracellular_positions.update(range(start["value"], end["value"] + 1))

    # Map protein positions to alignment columns
    is_extracellular = np.zeros(n_sites, dtype=np.float64)
    for i in range(n_sites):
        if col_to_protein_pos[i] in extracellular_positions:
            is_extracellular[i] = 1.0

    return is_extracellular
