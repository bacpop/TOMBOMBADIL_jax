import csv
import os
import tempfile
import unittest # for performing unit tests
import numpy as np
import jax.numpy as jnp

from tombombadil.sample import make_fn
from tombombadil.sample import evaluate_fixed_params
from tombombadil.sample import save_params
from tombombadil.sample import transforms
from tombombadil.sample import softplus_inverse
from tombombadil.__main__ import count_codons

# a test for calculating the likelihood (fn) for one codon
# run via python -m unittest -v test.test_fn.Testdiv
class Testdiv(unittest.TestCase):
        def testdiv(self):
            X = np.zeros((61,1))
            X[15,:] = 4
            X[47,:] = 19
            pi_test = np.array([1/61 for i in range(61)])
            log_pi, pimat, pimatinv, pimult = transforms(X, pi_test)
            mask = jnp.ones(1)

            fn = make_fn(pi_test, log_pi, pimat, pimatinv, pimult, X, mask)
            self.assertAlmostEqual(fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), 
                                    "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1), 
                                    "theta": softplus_inverse(0.5), "omega": jnp.array(softplus_inverse(0.5), dtype=jnp.float32)}), 
                                    jnp.array(-19.270576, dtype=jnp.float32), places=3)
            

# this is a test for the count_codons function (X, n_samples = count_codons(options.alignment))
# run via python -m unittest -v test.test_fn.Test_codon_count_matrix
class Test_codon_count_matrix(unittest.TestCase):
    #fasta_path = "porB3.carriage.noindels.txt"

    def setUp(self):
        self.fasta_path = "porB3.carriage.noindels.txt"

    def _compute_expected_matrix(self, fasta_path):
        """
        Reads a codon alignment in FASTA format, independently counts codons
        per alignment position, and compares results to my_func output.
        """

        # --- Generate full codon list ---
        bases = ["T", "C", "A", "G"]
        all_codons = [a + b + c for a in bases for b in bases for c in bases]

        # --- Remove stop codons ---
        stop_codons = {"TAA", "TAG", "TGA"}
        codon_list = [c for c in all_codons if c not in stop_codons]

        # Sanity check
        assert len(codon_list) == 61

        codon_index = {codon: i for i, codon in enumerate(codon_list)}

        # --- Read FASTA ---
        sequences = []
        with open(fasta_path) as f:
            seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seq:
                        sequences.append("".join(seq).upper())
                        seq = []
                else:
                    seq.append(line)
            if seq:
                sequences.append("".join(seq).upper())

        if not sequences:
            raise ValueError("No sequences found.")

        seq_length = len(sequences[0])
        if any(len(s) != seq_length for s in sequences):
            raise ValueError("Sequences are not aligned.")

        if seq_length % 3 != 0:
            raise ValueError("Alignment length not divisible by 3.")

        n_codons = seq_length // 3
        matrix = np.zeros((61, n_codons), dtype=int)

        # --- Count codons ---
        for seq in sequences:
            for pos in range(n_codons):
                codon = seq[pos*3:(pos+1)*3]

                # Skip gap-containing codons
                if "-" in codon:
                    continue

                # Skip stop codons
                if codon in stop_codons:
                    continue

                if codon in codon_index:
                    matrix[codon_index[codon], pos] += 1

        return matrix

    def test_codon_count_matrix(self):
        expected = self._compute_expected_matrix(self.fasta_path)
        observed, samples = count_codons(self.fasta_path)

        self.assertEqual(expected.shape, observed.shape)
        self.assertTrue((expected == observed).all())

class TestScalarOmegaOutput(unittest.TestCase):
    def test_save_params_writes_scalar_omega_only(self):
        params = {
            "alpha": jnp.array(softplus_inverse(1), dtype=jnp.float64),
            "beta": jnp.array(softplus_inverse(1), dtype=jnp.float64),
            "gamma": jnp.array(softplus_inverse(1), dtype=jnp.float64),
            "delta": jnp.array(softplus_inverse(1), dtype=jnp.float64),
            "epsilon": jnp.array(softplus_inverse(1), dtype=jnp.float64),
            "theta": jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
            "omega": jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        }
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "fit")
            save_params(stem, params)

            self.assertTrue(os.path.exists(stem + "_scalar.csv"))
            self.assertFalse(os.path.exists(stem + "_omega.csv"))

            with open(stem + "_scalar.csv", newline="") as f:
                rows = {row["variable"]: float(row["value"]) for row in csv.DictReader(f)}

        self.assertIn("omega", rows)
        self.assertAlmostEqual(rows["omega"], 0.5, places=6)


class TestDiagnosticObjective(unittest.TestCase):
    def test_fixed_param_sum_is_site_count_times_mean_without_priors(self):
        X = np.zeros((61, 2))
        X[15, :] = 4
        X[47, :] = 19
        pi_test = np.array([1 / 61 for i in range(61)])
        params = {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "delta": 1.0,
            "epsilon": 1.0,
            "eta": 1.0,
            "theta": 0.5,
            "omega": 0.5,
        }

        mean_value = evaluate_fixed_params(
            X, pi_test, params, aggregate="mean", prior_mode="none",
            eigen_jitter=False, omega_floor=False,
        )
        sum_value = evaluate_fixed_params(
            X, pi_test, params, aggregate="sum", prior_mode="none",
            eigen_jitter=False, omega_floor=False,
        )

        self.assertAlmostEqual(sum_value, 2 * mean_value, places=6)

    def test_fixed_eta_ignores_diagnostic_eta_parameter(self):
        X = np.zeros((61, 1))
        X[15, :] = 4
        X[47, :] = 19
        pi_test = np.array([1 / 61 for i in range(61)])
        params_eta_one = {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "delta": 1.0,
            "epsilon": 1.0,
            "eta": 1.0,
            "theta": 0.5,
            "omega": 0.5,
        }
        params_eta_two = dict(params_eta_one)
        params_eta_two["eta"] = 2.0

        eta_one = evaluate_fixed_params(
            X, pi_test, params_eta_one, estimate_eta=False,
            prior_mode="none", eigen_jitter=False, omega_floor=False,
        )
        eta_two = evaluate_fixed_params(
            X, pi_test, params_eta_two, estimate_eta=False,
            prior_mode="none", eigen_jitter=False, omega_floor=False,
        )

        self.assertAlmostEqual(eta_one, eta_two, places=6)


if __name__ == '__main__':
    unittest.main()
