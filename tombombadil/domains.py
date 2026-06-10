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
    return build_alignment_to_protein_map_for_length(alignment_path, len(ref_protein))


def build_alignment_to_protein_map_for_length(alignment_path, reference_protein_length):
    """Build an alignment-to-protein map using a known reference protein length."""
    ref_len = int(reference_protein_length)
    if ref_len <= 0:
        raise ValueError("Reference protein length must be positive")

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


def _impute_unknown_sites(is_extracellular, annotated_mask):
    """Impute unannotated alignment sites using nearest annotated neighbours.

    For each unannotated site, finds the nearest annotated site on the left and on
    the right (skipping over other unannotated sites):
      - Both neighbours extracellular     → impute as extracellular, include in regression
      - Both neighbours non-extracellular → impute as non-extracellular, include in regression
      - Neighbours disagree               → mark as NA, exclude from regression
      - No annotated neighbour on one side (edge) → mark as NA, exclude from regression

    Args:
        is_extracellular: float64 array, 1.0/0.0 for annotated sites (ignored for unknowns)
        annotated_mask: bool array, True where the site has a known domain annotation

    Returns:
        is_extracellular: updated float64 array (imputed values filled in; NA sites set to 0)
        is_imputed: bool array, True for sites that were imputed by this function
        regression_mask: float64 array, 1.0 where the site should be included in the regression
    """
    n = len(is_extracellular)
    result = is_extracellular.copy()
    is_imputed = np.zeros(n, dtype=bool)
    regression_mask = annotated_mask.astype(np.float64)

    for i in range(n):
        if annotated_mask[i]:
            continue  # already known

        # Nearest annotated neighbour on the left
        left_val = None
        for j in range(i - 1, -1, -1):
            if annotated_mask[j]:
                left_val = is_extracellular[j]
                break

        # Nearest annotated neighbour on the right
        right_val = None
        for j in range(i + 1, n):
            if annotated_mask[j]:
                right_val = is_extracellular[j]
                break

        if left_val is None or right_val is None:
            # Edge: cannot determine both neighbours → NA
            result[i] = 0.0
            regression_mask[i] = 0.0
        elif left_val == right_val:
            # Neighbours agree → impute
            result[i] = left_val
            is_imputed[i] = True
            regression_mask[i] = 1.0
        else:
            # Neighbours disagree → NA
            result[i] = 0.0
            regression_mask[i] = 0.0

    return result, is_imputed, regression_mask


def parse_domain_json(json_path, alignment_path, reference_protein_path, n_sites):
    """Parse a UniProt JSON file and return domain annotation arrays for all alignment sites.

    Sites with a known domain annotation are labelled directly. Unannotated sites
    (insertions relative to the reference, or protein positions not covered by any
    feature) are imputed from their nearest annotated neighbours where possible, or
    marked as NA and excluded from the regression.

    Args:
        json_path: path to UniProt JSON file
        alignment_path: path to codon DNA alignment (FASTA format)
        reference_protein_path: path to reference protein sequence (FASTA format)
        n_sites: number of alignment sites (codon positions), used for validation

    Returns:
        is_extracellular: float64 array of shape (n_sites,), 1=extracellular, 0=other/NA
        is_imputed: bool array of shape (n_sites,), True where the label was inferred
        regression_mask: float64 array of shape (n_sites,), 1=include in regression, 0=exclude (NA)
    """
    col_to_protein_pos = build_alignment_to_protein_map(alignment_path, reference_protein_path)

    if len(col_to_protein_pos) != n_sites:
        raise ValueError(
            f"Alignment has {len(col_to_protein_pos)} codon columns but n_sites={n_sites}"
        )

    with open(json_path) as f:
        data = json.load(f)

    extracellular_positions = set()
    all_annotated_positions = set()

    for feature in data.get("features", []):
        loc = feature.get("location", {})
        start = loc.get("start", {})
        end = loc.get("end", {})

        if start.get("modifier") != "EXACT" or end.get("modifier") != "EXACT":
            warnings.warn(
                f"Skipping non-EXACT feature at positions "
                f"{start.get('value')}-{end.get('value')}: {feature.get('description')!r}"
            )
            continue

        positions = set(range(start["value"], end["value"] + 1))
        all_annotated_positions.update(positions)

        if feature.get("description", "").lower() == "extracellular":
            extracellular_positions.update(positions)

    # Build initial annotation arrays: only sites with a known domain are annotated
    is_extracellular = np.zeros(n_sites, dtype=np.float64)
    annotated_mask = np.zeros(n_sites, dtype=bool)

    for i in range(n_sites):
        p = col_to_protein_pos[i]
        if p != -1 and p in all_annotated_positions:
            annotated_mask[i] = True
            if p in extracellular_positions:
                is_extracellular[i] = 1.0

    # Impute unknown sites from neighbours
    is_extracellular, is_imputed, regression_mask = _impute_unknown_sites(
        is_extracellular, annotated_mask
    )

    n_annotated = int(annotated_mask.sum())
    n_imputed = int(is_imputed.sum())
    n_na = int((regression_mask == 0).sum())
    n_extracellular = int(is_extracellular.sum())
    print(
        f"Domain annotation: {n_annotated} annotated, {n_imputed} imputed, "
        f"{n_na} excluded (NA) | {n_extracellular} extracellular total"
    )

    return is_extracellular, is_imputed, regression_mask
