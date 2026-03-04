import unittest # for performing unit tests
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import tombombadil
from tombombadil.sample import make_fn
from tombombadil.sample import transforms
from tombombadil.sample import softplus_inverse
from tombombadil.__main__ import count_codons

# a test for calculating the likelihood (fn) for one codon, correct value from Stan implementation
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
                                    "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(0.5), dtype=jnp.float32), jnp.size(X, axis=1))}), 
                                    jnp.array(-10.213031, dtype=jnp.float32), places=3)
            

# this is not a real test, just checking the likelihood values for a specific case (column 9 of the porB alignment)
# run via python -m unittest -v test.test_fn.Testlike
class Testlike(unittest.TestCase):
        def testlike(self):
            X = np.zeros((61,1))
            #X[21,:] = 5
            #X[22,:] = 18
            X[15,:] = 4
            X[47,:] = 19
            pi_test = np.array([1/61 for i in range(61)])
            log_pi, pimat, pimatinv, pimult = transforms(X, pi_test)
            mask = jnp.ones(1)

            fn = make_fn(pi_test, log_pi, pimat, pimatinv, pimult, X, mask)

            fn_i = np.zeros(20)
            fn_result = np.zeros(20)
            for i in range(20):
                print("i", i * 0.0001)
                #print(fn({"alpha": softplus_inverse(0.08288978), "beta": softplus_inverse(0.08277921), "gamma": softplus_inverse(0.07815452), 
                #                    "delta": softplus_inverse(0.08275475), "epsilon": softplus_inverse(0.06998631), "eta": softplus_inverse(3.1111848), 
                #                    "theta": softplus_inverse(2.203008), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))}))
                #print(fn({"alpha": softplus_inverse(0.8288978), "beta": softplus_inverse(0.8277921), "gamma": softplus_inverse(0.7815452), 
                #                    "delta": softplus_inverse(0.8275475), "epsilon": softplus_inverse(0.6998631), "eta": softplus_inverse(3.1111848), 
                #                    "theta": softplus_inverse(2.203008), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))}))
                print(fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), 
                                    "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1), 
                                    "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.0001), dtype=jnp.float32), jnp.size(X, axis=1))})) # param set which is like Ny98 and shows nans (replicated in Stan with omega=0.003)
                fn_result[i] = fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), 
                                    "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1.0), 
                                    "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.0001), dtype=jnp.float32), jnp.size(X, axis=1))})
                fn_i[i] = i * 0.0001
            plt.plot(fn_i, fn_result, 'o', color='black')
            plt.show()

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

# Tests for domain parsing and regression
# run via python -m unittest -v test.test_fn.TestDomainParsing
# run via python -m unittest -v test.test_fn.TestRegressionLikelihood
import jax
import jax.numpy as jnp
from tombombadil.domains import build_alignment_to_protein_map, parse_domain_json
from tombombadil.sample import regression_log_likelihood, make_fn, transforms, softplus_inverse

class TestDomainParsing(unittest.TestCase):
    alignment_path = "data/lamB_revtrans.fas.aln"
    reference_path = "data/lamB_reference.fas"
    domains_path = "data/lamB_domains.JSON"

    def _parse(self):
        from tombombadil.__main__ import count_codons
        X, _ = count_codons(self.alignment_path)
        return parse_domain_json(
            self.domains_path, self.alignment_path, self.reference_path, X.shape[1]
        )

    def test_alignment_to_protein_map_shape(self):
        col_to_protein_pos = build_alignment_to_protein_map(self.alignment_path, self.reference_path)
        self.assertEqual(len(col_to_protein_pos), 458)

    def test_alignment_to_protein_map_reference_length(self):
        col_to_protein_pos = build_alignment_to_protein_map(self.alignment_path, self.reference_path)
        n_real = (col_to_protein_pos != -1).sum()
        self.assertEqual(n_real, 446)

    def test_alignment_to_protein_map_positions_ascending(self):
        col_to_protein_pos = build_alignment_to_protein_map(self.alignment_path, self.reference_path)
        real_positions = col_to_protein_pos[col_to_protein_pos != -1]
        np.testing.assert_array_equal(real_positions, np.arange(1, 447))

    def test_returns_three_arrays(self):
        result = self._parse()
        self.assertEqual(len(result), 3)
        is_extracellular, is_imputed, regression_mask = result
        n = 458
        self.assertEqual(is_extracellular.shape, (n,))
        self.assertEqual(is_imputed.shape,       (n,))
        self.assertEqual(regression_mask.shape,  (n,))

    def test_is_extracellular_binary(self):
        is_extracellular, _, _ = self._parse()
        self.assertTrue(np.all((is_extracellular == 0) | (is_extracellular == 1)))

    def test_regression_mask_binary(self):
        _, _, regression_mask = self._parse()
        self.assertTrue(np.all((regression_mask == 0) | (regression_mask == 1)))

    def test_has_both_extracellular_and_other(self):
        is_extracellular, _, regression_mask = self._parse()
        included = regression_mask == 1
        self.assertGreater(is_extracellular[included].sum(), 0)
        self.assertGreater((1 - is_extracellular[included]).sum(), 0)

    def test_imputed_sites_included_in_regression(self):
        """All imputed sites must have regression_mask=1."""
        _, is_imputed, regression_mask = self._parse()
        self.assertTrue(np.all(regression_mask[is_imputed] == 1.0))

    def test_known_extracellular_region(self):
        """Protein positions 41-64 are extracellular in lamB (first extracellular loop)."""
        col_to_protein_pos = build_alignment_to_protein_map(self.alignment_path, self.reference_path)
        is_extracellular, _, _ = self._parse()
        extracellular_cols = np.where(
            (col_to_protein_pos >= 41) & (col_to_protein_pos <= 64)
        )[0]
        self.assertTrue(len(extracellular_cols) > 0)
        self.assertTrue(np.all(is_extracellular[extracellular_cols] == 1.0))

    def test_known_periplasmic_region(self):
        """Protein position 26 is Periplasmic in lamB — should not be extracellular."""
        col_to_protein_pos = build_alignment_to_protein_map(self.alignment_path, self.reference_path)
        is_extracellular, _, _ = self._parse()
        periplasmic_cols = np.where(col_to_protein_pos == 26)[0]
        self.assertTrue(len(periplasmic_cols) > 0)
        self.assertTrue(np.all(is_extracellular[periplasmic_cols] == 0.0))


class TestRegressionLikelihood(unittest.TestCase):
    _ALL_INCLUDED = jnp.ones(5, dtype=jnp.float64)

    def _make_params(self, omega_val, alpha_reg=0.0, beta_reg=0.0, log_sigma=0.0):
        return {
            "alpha":     jnp.array(softplus_inverse(1),         dtype=jnp.float64),
            "beta":      jnp.array(softplus_inverse(1),         dtype=jnp.float64),
            "gamma":     jnp.array(softplus_inverse(1),         dtype=jnp.float64),
            "delta":     jnp.array(softplus_inverse(1),         dtype=jnp.float64),
            "epsilon":   jnp.array(softplus_inverse(1),         dtype=jnp.float64),
            "eta":       jnp.array(softplus_inverse(1),         dtype=jnp.float64),
            "theta":     jnp.array(softplus_inverse(0.5),       dtype=jnp.float64),
            "omega":     jnp.repeat(jnp.array(softplus_inverse(omega_val), dtype=jnp.float64), 5),
            "alpha_reg": jnp.array(alpha_reg,  dtype=jnp.float64),
            "beta_reg":  jnp.array(beta_reg,   dtype=jnp.float64),
            "log_sigma": jnp.array(log_sigma,  dtype=jnp.float64),
        }

    def test_gradients_flow_through_regression_params(self):
        """jax.grad must produce non-zero gradients for alpha_reg, beta_reg, log_sigma."""
        is_extracellular = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        params = self._make_params(omega_val=0.5)

        def loss(p):
            return -regression_log_likelihood(p, is_extracellular, self._ALL_INCLUDED)

        grads = jax.grad(loss)(params)
        self.assertFalse(jnp.isnan(grads["alpha_reg"]))
        self.assertFalse(jnp.isnan(grads["beta_reg"]))
        self.assertFalse(jnp.isnan(grads["log_sigma"]))
        self.assertNotEqual(float(grads["alpha_reg"]), 0.0)
        self.assertNotEqual(float(grads["log_sigma"]), 0.0)

    def test_masked_sites_do_not_affect_gradient(self):
        """Sites with regression_mask=0 should not contribute to gradients."""
        is_extracellular = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        mask_all  = jnp.ones(5,  dtype=jnp.float64)
        mask_half = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
        # Non-zero beta_reg makes extracellular and non-extracellular sites have different
        # per-site likelihoods, so different masks produce different means.
        params = self._make_params(omega_val=0.5, beta_reg=1.0)

        ll_all  = regression_log_likelihood(params, is_extracellular, mask_all)
        ll_half = regression_log_likelihood(params, is_extracellular, mask_half)
        # Different masks → different likelihoods (last two sites are different domain type)
        self.assertNotAlmostEqual(float(ll_all), float(ll_half), places=4)

    def test_large_sigma_makes_regression_negligible(self):
        """With very large sigma, the gradient on omega should be smaller."""
        is_extracellular = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        params_large = self._make_params(omega_val=0.5, log_sigma=10.0)
        params_small = self._make_params(omega_val=0.5, log_sigma=0.0)

        def reg_ll(p): return regression_log_likelihood(p, is_extracellular, self._ALL_INCLUDED)
        grad_large = jax.grad(reg_ll)(params_large)["omega"]
        grad_small = jax.grad(reg_ll)(params_small)["omega"]
        self.assertLess(
            float(jnp.max(jnp.abs(grad_large))),
            float(jnp.max(jnp.abs(grad_small)))
        )

    def test_small_sigma_pulls_omega_toward_prediction(self):
        """With very small sigma, regression strongly constrains omega to the prediction."""
        is_extracellular = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        params = self._make_params(omega_val=0.5, alpha_reg=0.0, beta_reg=0.0, log_sigma=-5.0)

        def reg_ll(p): return regression_log_likelihood(p, is_extracellular, self._ALL_INCLUDED)
        grads = jax.grad(reg_ll)(params)
        self.assertTrue(jnp.all(grads["omega"] > 0))


if __name__ == '__main__':
    unittest.main()