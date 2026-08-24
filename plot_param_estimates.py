#!/usr/bin/env python
"""Plot TOMBOMBADIL MAP parameter estimates from saved CSV files.

Reads CSV pairs written by ``tombombadil.sample.save_params``:

    STEM_omega.csv   with columns: site, omega_map, variant
    STEM_scalar.csv  with columns: variable, value

In stem mode, produces two PNG files by default:

    STEM_omega_plot.png
    STEM_scalar_plot.png

In folder mode, writes one pair of PNG files per complete CSV pair found in
the input folder.
"""

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from tombombadil.domains import build_alignment_to_protein_map_for_length


_PREFERRED_SCALAR_ORDER = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "eta",
    "theta",
]

_DOMAIN_STYLES = {
    "O": ("tomato", "O"),
    "I": ("steelblue", "I"),
    "L": ("darkorange", "L"),
    "p": ("seagreen", "p"),
}
_UNKNOWN_DOMAIN = "?"
_ALIGNMENT_EXTENSIONS = (
    ".fas.aln",
    ".fasta.aln",
    ".fa.aln",
    ".aln",
    ".fasta",
    ".fas",
    ".fa",
    ".fna",
    ".txt",
)
_ALIGNMENT_SUFFIXES = ("_codon_aligned",)
_UNIREF_PREFIX_RE = re.compile(r"^UniRef\d+_", re.IGNORECASE)


def _require_columns(path, names, required):
    missing = [name for name in required if name not in names]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{path} is missing required column(s): {missing_str}")


def _unique(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _strip_uniref_prefix(name):
    return _UNIREF_PREFIX_RE.sub("", name)


def _stem_match_keys(name):
    base = os.path.basename(str(name))
    return _unique([base, _strip_uniref_prefix(base)])


def _record_match_keys(header):
    if header is None:
        return []

    header = header.strip()
    first_token = header.split()[0] if header.split() else header
    parts = [header, first_token]
    if "|" in first_token:
        parts.extend(first_token.split("|"))

    keys = []
    for part in parts:
        part = part.strip()
        keys.extend([part, _strip_uniref_prefix(part)])
    return _unique(keys)


def _read_fasta_like_records(path):
    """Read one or more FASTA-like records from a sequence or annotation file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    records = []
    header = None
    sequence = []
    saw_header = False

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header or sequence:
                    records.append((header, "".join(sequence)))
                header = line[1:].strip()
                sequence = []
                saw_header = True
            else:
                sequence.append("".join(line.split()))

    if saw_header or sequence:
        records.append((header, "".join(sequence)))

    if not records:
        raise ValueError(f"No sequence found in {path}")
    for header, sequence in records:
        if sequence == "":
            label = header if header is not None else "<no header>"
            raise ValueError(f"No sequence found for record {label!r} in {path}")
    return records


def _select_record_sequence(records, path, record_key=None, role="sequence"):
    if record_key is not None:
        desired_keys = _stem_match_keys(record_key)
        for desired_key in desired_keys:
            for header, sequence in records:
                if desired_key in _record_match_keys(header):
                    return sequence

    if len(records) == 1:
        return records[0][1]

    if record_key is None:
        raise ValueError(
            f"{path} contains multiple {role} records; a record name is required"
        )

    tried = ", ".join(_stem_match_keys(record_key))
    raise ValueError(
        f"Could not find {role} record matching {record_key!r} in {path} "
        f"(tried: {tried})"
    )


def _strip_alignment_extension(filename):
    root = filename
    if root.endswith(".gz"):
        root = root[: -len(".gz")]
    for extension in _ALIGNMENT_EXTENSIONS:
        if root.endswith(extension):
            return root[: -len(extension)]
    return os.path.splitext(root)[0]


def _alignment_match_keys(name):
    keys = _stem_match_keys(name)
    for suffix in _ALIGNMENT_SUFFIXES:
        if name.endswith(suffix):
            keys.extend(_stem_match_keys(name[: -len(suffix)]))
    return _unique(keys)


def _resolve_alignment_path(alignment_arg, stem_name=None):
    if alignment_arg is None or not os.path.isdir(alignment_arg):
        return alignment_arg

    if stem_name is None:
        raise ValueError("--alignment is a folder, but no stem name was provided")

    candidate_names = _stem_match_keys(stem_name)
    for name in candidate_names:
        for suffix in ("",) + _ALIGNMENT_SUFFIXES:
            for extension in ("",) + _ALIGNMENT_EXTENSIONS:
                candidate = os.path.join(alignment_arg, name + suffix + extension)
                if os.path.isfile(candidate):
                    return candidate

    matches = []
    desired_keys = set(candidate_names)
    for candidate in sorted(glob.glob(os.path.join(alignment_arg, "*"))):
        if not os.path.isfile(candidate):
            continue
        candidate_root = _strip_alignment_extension(os.path.basename(candidate))
        candidate_keys = set(_alignment_match_keys(candidate_root))
        if desired_keys & candidate_keys:
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        match_list = ", ".join(matches)
        raise ValueError(
            f"Multiple alignments in {alignment_arg} match {stem_name!r}: {match_list}"
        )

    tried = ", ".join(candidate_names)
    raise FileNotFoundError(
        f"Could not find an alignment in {alignment_arg} matching {stem_name!r} "
        f"(tried: {tried})"
    )


def load_omega_csv(path):
    """Load per-site omega estimates from an omega CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Omega CSV not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError(f"{path} does not look like a named-column CSV")

    _require_columns(path, data.dtype.names, ["site", "omega_map"])

    sites = np.atleast_1d(data["site"]).astype(int)
    omega = np.atleast_1d(data["omega_map"]).astype(np.float64)
    variant = None
    if "variant" in data.dtype.names:
        variant = np.atleast_1d(data["variant"]).astype(int)
        if len(variant) != len(omega):
            raise ValueError(f"{path} has mismatched omega_map and variant lengths")

    if len(sites) != len(omega):
        raise ValueError(f"{path} has mismatched site and omega_map lengths")

    return sites, omega, variant


def load_domain_annotations(
    annotation_path,
    n_sites,
    alignment_path=None,
    reference_path=None,
    record_key=None,
):
    """Load per-site domain annotations from a reference-matched annotation string.

    If alignment/reference are supplied, the annotation is mapped from reference
    protein positions to codon-alignment columns. Alignment insertions relative
    to the reference are assigned an unknown annotation.
    """
    annotation = _select_record_sequence(
        _read_fasta_like_records(annotation_path),
        annotation_path,
        record_key=record_key,
        role="domain annotation",
    )

    if alignment_path is None and reference_path is None:
        if len(annotation) != n_sites:
            raise ValueError(
                f"{annotation_path} has {len(annotation)} annotations, but the omega CSV "
                f"has {n_sites} sites. Provide --alignment and --reference-protein to map "
                "reference positions to alignment columns."
            )
        return np.array(list(annotation), dtype=object)

    if alignment_path is None or reference_path is None:
        raise ValueError("--alignment and --reference-protein must be provided together")

    reference = _select_record_sequence(
        _read_fasta_like_records(reference_path),
        reference_path,
        record_key=record_key,
        role="reference protein",
    )
    if len(annotation) != len(reference):
        raise ValueError(
            f"{annotation_path} has {len(annotation)} annotations, but {reference_path} "
            f"has {len(reference)} amino acids"
        )

    col_to_protein_pos = build_alignment_to_protein_map_for_length(
        alignment_path, len(reference)
    )
    if len(col_to_protein_pos) != n_sites:
        raise ValueError(
            f"{alignment_path} has {len(col_to_protein_pos)} codon columns, but the omega "
            f"CSV has {n_sites} sites"
        )

    per_site = np.full(n_sites, _UNKNOWN_DOMAIN, dtype=object)
    for i, protein_pos in enumerate(col_to_protein_pos):
        if protein_pos != -1:
            per_site[i] = annotation[int(protein_pos) - 1]

    return per_site


def load_scalar_csv(path, include_eta=True):
    """Load scalar parameters from a scalar CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scalar CSV not found: {path}")

    scalars = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not look like a named-column CSV")
        _require_columns(path, reader.fieldnames, ["variable", "value"])
        for row in reader:
            variable = row["variable"]
            if variable == "":
                continue
            scalars[variable] = float(row["value"])

    if include_eta and "eta" not in scalars:
        scalars["eta"] = 1.0

    return scalars


def find_csv_pairs(folder):
    """Find complete *_omega.csv / *_scalar.csv pairs in a folder."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    pairs = []
    omega_paths = sorted(glob.glob(os.path.join(folder, "*_omega.csv")))
    for omega_path in omega_paths:
        stem = omega_path[: -len("_omega.csv")]
        scalar_path = stem + "_scalar.csv"
        if not os.path.exists(scalar_path):
            print(f"Skipping {omega_path}: matching scalar CSV not found")
            continue
        pairs.append((stem, omega_path, scalar_path))

    if not pairs:
        raise FileNotFoundError(f"No complete *_omega.csv / *_scalar.csv pairs found in {folder}")

    return pairs


def ordered_scalar_names(scalars):
    """Return scalar names in the standard order, followed by extras."""
    preferred = [name for name in _PREFERRED_SCALAR_ORDER if name in scalars]
    extras = sorted(name for name in scalars if name not in _PREFERRED_SCALAR_ORDER)
    return preferred + extras


def plot_omega(sites, omega, variant=None, domain_annotations=None, log_scale=True):
    """Create a per-site omega scatter plot."""
    fig, ax = plt.subplots(figsize=(16, 4))

    if domain_annotations is not None:
        domains = np.asarray(domain_annotations, dtype=object)
        if len(domains) != len(omega):
            raise ValueError("Domain annotation length does not match omega length")

        plotted = np.zeros(len(omega), dtype=bool)
        for code, (colour, label) in _DOMAIN_STYLES.items():
            mask = domains == code
            if mask.any():
                ax.scatter(
                    sites[mask],
                    omega[mask],
                    color=colour,
                    s=9,
                    alpha=0.75,
                    linewidths=0,
                    label=label,
                    zorder=3,
                )
                plotted |= mask

        unknown = ~plotted
        if unknown.any():
            ax.scatter(
                sites[unknown],
                omega[unknown],
                color="lightgrey",
                s=7,
                alpha=0.6,
                linewidths=0,
                label="Unknown / insertion",
                zorder=2,
            )
    elif variant is None:
        ax.scatter(
            sites,
            omega,
            color="black",
            s=8,
            alpha=0.65,
            linewidths=0,
            label="Omega estimate",
            zorder=3,
        )
    else:
        invariant = variant == 0
        variable = variant != 0
        if invariant.any():
            ax.scatter(
                sites[invariant],
                omega[invariant],
                color="lightgrey",
                s=7,
                alpha=0.7,
                linewidths=0,
                label="Invariant site",
                zorder=2,
            )
        if variable.any():
            ax.scatter(
                sites[variable],
                omega[variable],
                color="black",
                s=9,
                alpha=0.75,
                linewidths=0,
                label="Variable site",
                zorder=3,
            )

    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label="omega = 1",
        zorder=4,
    )
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("omega (dN/dS, log scale)")
    else:
        ax.set_ylabel("omega (dN/dS)")
    ax.set_xlabel("Alignment site")
    ax.set_title("Per-site omega estimates")
    ax.set_xlim(sites.min() - 1, sites.max() + 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_omega_by_domain(omega, domain_annotations, log_scale=True):
    """Create a jittered per-site omega scatter plot grouped by domain."""
    omega = np.asarray(omega, dtype=np.float64)
    domains = np.asarray(domain_annotations, dtype=object)
    if len(domains) != len(omega):
        raise ValueError("Domain annotation length does not match omega length")

    category_order = list(_DOMAIN_STYLES) + [_UNKNOWN_DOMAIN]
    observed = [category for category in category_order if np.any(domains == category)]
    nonstandard = [
        category
        for category in _unique(domains.tolist())
        if category not in category_order
    ]
    categories = observed + nonstandard
    if not categories:
        raise ValueError("No domain annotations available")

    fig, ax = plt.subplots(figsize=(max(6.5, len(categories) * 1.25), 5))
    for x, category in enumerate(categories):
        mask = domains == category
        values = omega[mask]
        count = len(values)
        if count == 1:
            jitter = np.zeros(1)
        else:
            jitter = np.linspace(-0.16, 0.16, count)
        colour, label = _DOMAIN_STYLES.get(category, ("lightgrey", str(category)))
        ax.scatter(
            np.full(count, x, dtype=np.float64) + jitter,
            values,
            color=colour,
            s=12,
            alpha=0.7,
            linewidths=0,
            label=label,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.1, alpha=0.8,
               label="omega = 1")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel("Domain")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("omega (dN/dS, log scale)")
    else:
        ax.set_ylabel("omega (dN/dS)")
    ax.set_title("Per-site omega estimates by domain")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_omega_by_domain_distribution(omega, domain_annotations, log_scale=True):
    """Create domain-grouped omega boxplots with overlaid site estimates."""
    omega = np.asarray(omega, dtype=np.float64)
    domains = np.asarray(domain_annotations, dtype=object)
    if len(domains) != len(omega):
        raise ValueError("Domain annotation length does not match omega length")

    category_order = list(_DOMAIN_STYLES) + [_UNKNOWN_DOMAIN]
    observed = [category for category in category_order if np.any(domains == category)]
    nonstandard = [
        category
        for category in _unique(domains.tolist())
        if category not in category_order
    ]
    categories = observed + nonstandard
    if not categories:
        raise ValueError("No domain annotations available")

    values_by_category = [omega[domains == category] for category in categories]
    positions = np.arange(1, len(categories) + 1)
    fig, ax = plt.subplots(figsize=(max(6.5, len(categories) * 1.25), 5))
    box = ax.boxplot(
        values_by_category,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
    )

    for patch, category in zip(box["boxes"], categories):
        colour = _DOMAIN_STYLES.get(category, ("lightgrey", str(category)))[0]
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    for x, category in zip(positions, categories):
        values = omega[domains == category]
        count = len(values)
        jitter = np.zeros(1) if count == 1 else np.linspace(-0.16, 0.16, count)
        colour = _DOMAIN_STYLES.get(category, ("lightgrey", str(category)))[0]
        ax.scatter(
            np.full(count, x, dtype=np.float64) + jitter,
            values,
            color=colour,
            edgecolors="black",
            linewidths=0.25,
            s=12,
            alpha=0.7,
            zorder=3,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.1, alpha=0.8,
               label="omega = 1")
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Domain")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("omega (dN/dS, log scale)")
    else:
        ax.set_ylabel("omega (dN/dS)")
    ax.set_title("Omega estimate distributions by domain")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_omega_distribution_by_domain(omega, domain_annotations, log_scale=True):
    """Create offset, shared-bin omega histograms grouped by domain."""
    omega = np.asarray(omega, dtype=np.float64)
    domains = np.asarray(domain_annotations, dtype=object)
    if len(domains) != len(omega):
        raise ValueError("Domain annotation length does not match omega length")
    if len(omega) == 0:
        raise ValueError("No omega estimates available")

    category_order = list(_DOMAIN_STYLES) + [_UNKNOWN_DOMAIN]
    nonstandard = [
        category
        for category in _unique(domains.tolist())
        if category not in category_order
    ]
    categories = category_order + nonstandard
    bins = np.histogram_bin_edges(omega, bins="auto")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bin_widths = np.diff(bins)
    # Make each category bar fill its bin completely, with only a small offset
    # between categories so each colored distribution remains continuous.
    bar_widths = bin_widths
    bin_centres = (bins[:-1] + bins[1:]) / 2

    for category_index, category in enumerate(categories):
        values = omega[domains == category]
        counts, _ = np.histogram(values, bins=bins)
        colour, label = _DOMAIN_STYLES.get(category, ("lightgrey", str(category)))
        offset = (
            category_index - (len(categories) - 1) / 2
        ) * bin_widths * 0.025
        ax.bar(
            bin_centres + offset,
            counts,
            width=bar_widths,
            color=colour,
            alpha=0.5,
            edgecolor=colour,
            linewidth=0.6,
            label=label,
            align="center",
            zorder=2 + category_index,
        )

    ax.axvline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label="omega = 1",
    )
    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("omega (dN/dS, log scale)")
    else:
        ax.set_xlabel("omega (dN/dS)")
    ax.set_ylabel("Number of sites")
    ax.set_title("Omega estimate distributions by domain")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_scalars(scalars):
    """Create a scalar-parameter dot plot."""
    names = ordered_scalar_names(scalars)
    values = np.array([scalars[name] for name in names], dtype=np.float64)
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.5, len(names) * 0.55 + 1.5))
    ax.scatter(
        values,
        y,
        color="black",
        s=50,
        marker="D",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    if np.all(values > 0):
        ax.set_xscale("log")
        ax.set_xlabel("Parameter value (log scale)")
    else:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("Parameter value (signed log scale)")
    ax.set_title("Scalar parameter estimates")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    return fig


def write_plots(
    omega_path,
    scalar_path,
    omega_out,
    scalar_out,
    domain_annotation_path=None,
    alignment_path=None,
    reference_path=None,
    record_key=None,
    log_omega=False,
    include_eta=True,
    domain_omega_out=None,
    domain_distribution_out=None,
    domain_histogram_out=None,
):
    """Load one CSV pair and write its requested plot PNGs."""
    sites, omega, variant = load_omega_csv(omega_path)
    scalars = load_scalar_csv(scalar_path, include_eta=include_eta)
    domain_annotations = None
    if domain_annotation_path is not None:
        domain_annotations = load_domain_annotations(
            domain_annotation_path,
            len(omega),
            alignment_path=alignment_path,
            reference_path=reference_path,
            record_key=record_key,
        )

    omega_fig = plot_omega(
        sites,
        omega,
        variant,
        domain_annotations=domain_annotations,
        log_scale=log_omega,
    )
    omega_fig.savefig(omega_out, dpi=300, bbox_inches="tight")
    plt.close(omega_fig)

    if domain_annotations is not None and domain_omega_out is not None:
        domain_fig = plot_omega_by_domain(
            omega,
            domain_annotations,
            log_scale=log_omega,
        )
        domain_fig.savefig(domain_omega_out, dpi=300, bbox_inches="tight")
        plt.close(domain_fig)
        print(f"Wrote domain omega plot: {domain_omega_out}")

    if domain_annotations is not None and domain_distribution_out is not None:
        distribution_fig = plot_omega_by_domain_distribution(
            omega,
            domain_annotations,
            log_scale=log_omega,
        )
        distribution_fig.savefig(domain_distribution_out, dpi=300, bbox_inches="tight")
        plt.close(distribution_fig)
        print(f"Wrote domain distribution plot: {domain_distribution_out}")

    if domain_annotations is not None and domain_histogram_out is not None:
        histogram_fig = plot_omega_distribution_by_domain(
            omega,
            domain_annotations,
            log_scale=log_omega,
        )
        histogram_fig.savefig(domain_histogram_out, dpi=300, bbox_inches="tight")
        plt.close(histogram_fig)
        print(f"Wrote domain histogram plot: {domain_histogram_out}")

    scalar_fig = plot_scalars(scalars)
    scalar_fig.savefig(scalar_out, dpi=300, bbox_inches="tight")
    plt.close(scalar_fig)

    print(f"Wrote omega plot: {omega_out}")
    print(f"Wrote scalar plot: {scalar_out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TOMBOMBADIL omega and scalar parameter estimates from CSV files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--stem",
        help="Input stem. Reads STEM_omega.csv and STEM_scalar.csv.",
    )
    source.add_argument(
        "--folder",
        help="Input folder. Plots every complete *_omega.csv / *_scalar.csv pair in the folder.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output prefix for --stem mode, or output folder for --folder mode. "
            "Defaults to the input stem or folder."
        ),
    )
    parser.add_argument(
        "--domain-annotations",
        default=None,
        help=(
            "FASTA-like text file with one BOCTOPUS-style domain annotation "
            "character per reference amino acid. In --folder mode this may contain "
            "multiple named records."
        ),
    )
    parser.add_argument(
        "--reference-protein",
        default=None,
        help=(
            "Reference protein FASTA matched to --domain-annotations. In --folder "
            "mode this may contain multiple records; record names are matched to "
            "input stems after stripping a UniRef90_ prefix."
        ),
    )
    parser.add_argument(
        "--alignment",
        default=None,
        help=(
            "Codon alignment FASTA, or an alignment folder in --folder mode, used "
            "to map reference annotations to omega sites."
        ),
    )

    omega_scale = parser.add_mutually_exclusive_group()
    omega_scale.add_argument(
        "--log-omega",
        dest="log_omega",
        action="store_true",
        default=False,
        help="Plot omega on a log y-axis.",
    )
    omega_scale.add_argument(
        "--linear-omega",
        dest="log_omega",
        action="store_false",
        help="Plot omega on a linear y-axis (default).",
    )

    parser.add_argument(
        "--no-fixed-eta",
        dest="include_eta",
        action="store_false",
        default=True,
        help="Do not add fixed eta=1.0 when eta is absent from the scalar CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.domain_annotations is None and (
        args.alignment is not None or args.reference_protein is not None
    ):
        raise ValueError("--alignment and --reference-protein require --domain-annotations")
    if args.domain_annotations is not None and (
        (args.alignment is None) != (args.reference_protein is None)
    ):
        raise ValueError("--alignment and --reference-protein must be provided together")

    if args.folder is not None:
        output_folder = args.output_prefix or args.folder
        os.makedirs(output_folder, exist_ok=True)
        for stem, omega_path, scalar_path in find_csv_pairs(args.folder):
            stem_name = os.path.basename(stem)
            alignment_path = _resolve_alignment_path(args.alignment, stem_name)
            omega_out = os.path.join(output_folder, stem_name + "_omega_plot.png")
            domain_omega_out = os.path.join(
                output_folder, stem_name + "_omega_by_domain_plot.png"
            )
            domain_distribution_out = os.path.join(
                output_folder, stem_name + "_omega_by_domain_distribution_plot.png"
            )
            domain_histogram_out = os.path.join(
                output_folder, stem_name + "_omega_distribution_by_domain_plot.png"
            )
            scalar_out = os.path.join(output_folder, stem_name + "_scalar_plot.png")
            write_plots(
                omega_path,
                scalar_path,
                omega_out,
                scalar_out,
                domain_omega_out=domain_omega_out,
                domain_distribution_out=domain_distribution_out,
                domain_histogram_out=domain_histogram_out,
                domain_annotation_path=args.domain_annotations,
                alignment_path=alignment_path,
                reference_path=args.reference_protein,
                record_key=stem_name,
                log_omega=args.log_omega,
                include_eta=args.include_eta,
            )
    else:
        output_prefix = args.output_prefix or args.stem
        stem_name = os.path.basename(args.stem)
        omega_path = args.stem + "_omega.csv"
        scalar_path = args.stem + "_scalar.csv"
        omega_out = output_prefix + "_omega_plot.png"
        domain_omega_out = output_prefix + "_omega_by_domain_plot.png"
        domain_distribution_out = (
            output_prefix + "_omega_by_domain_distribution_plot.png"
        )
        domain_histogram_out = output_prefix + "_omega_distribution_by_domain_plot.png"
        scalar_out = output_prefix + "_scalar_plot.png"
        write_plots(
            omega_path,
            scalar_path,
            omega_out,
            scalar_out,
            domain_omega_out=domain_omega_out,
            domain_distribution_out=domain_distribution_out,
            domain_histogram_out=domain_histogram_out,
            domain_annotation_path=args.domain_annotations,
            alignment_path=args.alignment,
            reference_path=args.reference_protein,
            record_key=stem_name,
            log_omega=args.log_omega,
            include_eta=args.include_eta,
        )


if __name__ == "__main__":
    main()
