#!/usr/bin/env python
"""Plot per-site codon usage for diverse sites, split by synonymous/nonsynonymous change.

For each alignment site that shows diversity (more than one codon observed),
produces a bar chart showing counts of all 61 sense codons, coloured by:
  - grey       : the dominant (most prevalent) codon
  - steelblue  : synonymous change relative to the dominant codon
  - tomato     : nonsynonymous change relative to the dominant codon
  - lightgrey  : codon not observed at this site (count == 0)

All plots use the same fixed x-axis (all 61 codons in TCAG order) so that
sites can be compared directly. Output is a multi-page PDF.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from tombombadil.__main__ import count_codons
from tombombadil.genetic_code import AA_LIST, CODON_LIST


def is_nonsyn(i, j):
    """Return True if codons at indices i and j encode different amino acids."""
    return AA_LIST[i] != AA_LIST[j]


def plot_site(ax, counts, dominant, codon_list, site_idx):
    """Draw one bar chart onto *ax* for a single alignment site."""
    n = len(codon_list)
    x = np.arange(n)

    # Classify each codon
    colours = []
    for i, count in enumerate(counts):
        if count == 0:
            colours.append("lightgrey")
        elif i == dominant:
            colours.append("dimgrey")
        elif is_nonsyn(dominant, i):
            colours.append("tomato")
        else:
            colours.append("steelblue")

    ax.bar(x, counts, color=colours, edgecolor="none", width=0.8)

    # Shade the full background column for any present non-dominant codon so
    # that even very small counts are visible at a glance.
    for i, (count, colour) in enumerate(zip(counts, colours)):
        if count > 0 and colour in ("steelblue", "tomato"):
            ax.axvspan(i - 0.4, i + 0.4, color=colour, alpha=0.15, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(codon_list, rotation=90, fontsize=5)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log₁₀)")
    ax.set_title(f"Site {site_idx + 1}  —  dominant: {codon_list[dominant]}", fontsize=9)
    ax.set_xlim(-0.5, n - 0.5)

    # Legend (only show categories that are present)
    legend_elements = [Patch(facecolor="dimgrey",   label="Dominant")]
    if any(c == "steelblue" for c in colours):
        legend_elements.append(Patch(facecolor="steelblue", label="Synonymous"))
    if any(c == "tomato" for c in colours):
        legend_elements.append(Patch(facecolor="tomato",    label="Nonsynonymous"))
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-site codon diversity (syn/nonsyn) for all diverse alignment sites."
    )
    parser.add_argument("--alignment", required=True,
                        help="Codon alignment FASTA file (plain or gzipped)")
    parser.add_argument("--output", default="codon_diversity.pdf",
                        help="Output PDF file (default: codon_diversity.pdf)")
    args = parser.parse_args()

    print(f"Reading alignment: {args.alignment}")
    X, n_samples = count_codons(args.alignment)
    n_sites = X.shape[1]
    print(f"  {n_samples} sequences, {n_sites} codon sites")

    # Identify diverse sites
    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    diverse_sites = np.where(col_max != col_sum)[0]
    print(f"  {len(diverse_sites)} diverse sites (out of {n_sites})")

    # Further filter to sites with at least one nonsynonymous observation
    nonsyn_sites = []
    for site in diverse_sites:
        counts = X[:, site]
        dominant = int(np.argmax(counts))
        has_nonsyn = any(
            counts[i] > 0 and i != dominant and is_nonsyn(dominant, i)
            for i in range(61)
        )
        if has_nonsyn:
            nonsyn_sites.append(site)
    print(f"  {len(nonsyn_sites)} sites with at least one nonsynonymous mutation")

    print(f"Writing plots to: {args.output}")
    with PdfPages(args.output) as pdf:
        for site in nonsyn_sites:
            counts = X[:, site]
            dominant = int(np.argmax(counts))

            fig, ax = plt.subplots(figsize=(14, 3.5))
            plot_site(ax, counts, dominant, CODON_LIST, site)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
