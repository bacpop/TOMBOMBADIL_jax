import os
import tempfile
import unittest

import numpy as np

from plot_param_estimates import _resolve_alignment_path, load_domain_annotations


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
