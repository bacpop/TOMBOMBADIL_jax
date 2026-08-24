import os
import tempfile
import unittest

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_param_estimates import (
    _resolve_alignment_path,
    load_domain_annotations,
    plot_omega_by_domain,
    plot_omega_by_domain_distribution,
    plot_omega_distribution_by_domain,
    write_plots,
)


class TestOmegaByDomainPlot(unittest.TestCase):
    def test_uses_domain_order_and_includes_unknowns(self):
        fig = plot_omega_by_domain(
            np.array([0.5, 2.0, 1.0, 4.0]),
            np.array(["I", "O", "?", "p"], dtype=object),
            log_scale=False,
        )

        self.assertEqual(
            [tick.get_text() for tick in fig.axes[0].get_xticklabels()],
            ["O", "I", "p", "?"],
        )
        self.assertEqual(fig.axes[0].get_xlabel(), "Domain")
        self.assertEqual(fig.axes[0].get_ylabel(), "omega (dN/dS)")
        plt.close(fig)

    def test_distribution_plot_uses_same_domain_order(self):
        fig = plot_omega_by_domain_distribution(
            np.array([0.5, 2.0, 1.0, 4.0, 0.8]),
            np.array(["I", "O", "?", "p", "O"], dtype=object),
            log_scale=False,
        )

        self.assertEqual(
            [tick.get_text() for tick in fig.axes[0].get_xticklabels()],
            ["O", "I", "p", "?"],
        )
        self.assertEqual(fig.axes[0].get_title(), "Omega estimate distributions by domain")
        plt.close(fig)

    def test_histogram_plot_uses_omega_x_axis_and_category_legend(self):
        fig = plot_omega_distribution_by_domain(
            np.array([0.5, 0.8, 1.5, 2.0, 4.0]),
            np.array(["I", "O", "O", "p", "?"], dtype=object),
            log_scale=True,
        )
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_xlabel(), "omega (dN/dS, log scale)")
        self.assertEqual(ax.get_ylabel(), "Number of sites")
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(legend_labels[:5], ["O", "I", "L", "p", "?"])
        plt.close(fig)

    def test_writes_domain_plot_when_annotations_are_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "fit")
            with open(stem + "_omega.csv", "w") as f:
                f.write("site,omega_map,variant\n1,0.5,0\n2,2.0,1\n")
            with open(stem + "_scalar.csv", "w") as f:
                f.write("variable,value\nalpha,1.0\n")
            annotation_path = os.path.join(tmp, "domains.fasta")
            with open(annotation_path, "w") as f:
                f.write(">fit\nIO\n")

            domain_out = os.path.join(tmp, "fit_omega_by_domain_plot.png")
            distribution_out = os.path.join(
                tmp, "fit_omega_by_domain_distribution_plot.png"
            )
            histogram_out = os.path.join(
                tmp, "fit_omega_distribution_by_domain_plot.png"
            )
            write_plots(
                stem + "_omega.csv",
                stem + "_scalar.csv",
                stem + "_omega_plot.png",
                stem + "_scalar_plot.png",
                domain_omega_out=domain_out,
                domain_distribution_out=distribution_out,
                domain_histogram_out=histogram_out,
                domain_annotation_path=annotation_path,
            )

            self.assertTrue(os.path.exists(domain_out))
            self.assertGreater(os.path.getsize(domain_out), 0)
            self.assertTrue(os.path.exists(distribution_out))
            self.assertGreater(os.path.getsize(distribution_out), 0)
            self.assertTrue(os.path.exists(histogram_out))
            self.assertGreater(os.path.getsize(histogram_out), 0)


class TestFolderDomainAnnotations(unittest.TestCase):
    def test_matches_uniref_stem_to_stripped_multifasta_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            alignment_dir = os.path.join(tmp, "alignments")
            os.mkdir(alignment_dir)

            alignment_path = os.path.join(
                alignment_dir, "UniRef90_xyz_codon_aligned.fasta"
            )
            with open(alignment_path, "w") as f:
                f.write(">seq1\nATGAAAACC\n")

            reference_path = os.path.join(tmp, "references.fasta")
            with open(reference_path, "w") as f:
                f.write(">abc\nMMMM\n>xyz\nMKT\n")

            annotation_path = os.path.join(tmp, "domains.txt")
            with open(annotation_path, "w") as f:
                f.write(">abc\nLLLL\n>xyz\nOIp\n")

            resolved = _resolve_alignment_path(alignment_dir, "UniRef90_xyz")
            self.assertEqual(resolved, alignment_path)

            annotations = load_domain_annotations(
                annotation_path,
                3,
                alignment_path=resolved,
                reference_path=reference_path,
                record_key="UniRef90_xyz",
            )

        np.testing.assert_array_equal(
            annotations, np.array(["O", "I", "p"], dtype=object)
        )

    def test_alignment_folder_can_match_stripped_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            alignment_path = os.path.join(tmp, "xyz.aln")
            with open(alignment_path, "w") as f:
                f.write(">seq1\nATGAAAACC\n")

            resolved = _resolve_alignment_path(tmp, "UniRef90_xyz")

        self.assertEqual(resolved, alignment_path)

    def test_alignment_folder_can_match_stripped_name_with_codon_aligned_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            alignment_path = os.path.join(tmp, "xyz_codon_aligned.fasta")
            with open(alignment_path, "w") as f:
                f.write(">seq1\nATGAAAACC\n")

            resolved = _resolve_alignment_path(tmp, "UniRef90_xyz")

        self.assertEqual(resolved, alignment_path)


if __name__ == "__main__":
    unittest.main()
