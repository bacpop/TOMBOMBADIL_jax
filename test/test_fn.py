import csv
import os
import sys
import tempfile
import unittest # for performing unit tests
from unittest import mock
import numpy as np
import jax
import jax.numpy as jnp
import optax

from tombombadil.__main__ import configure_jax_for_options
from tombombadil.__main__ import estimate_pi_from_counts
from tombombadil.__main__ import get_options
from tombombadil.sample import make_fn
from tombombadil.sample import _optimize_params
from tombombadil.sample import evaluate_fixed_params
from tombombadil.sample import run_nuts_sampler
from tombombadil.sample import save_params
from tombombadil.sample import save_posterior_outputs
from tombombadil.sample import summarize_posterior_samples
from tombombadil.sample import transforms
from tombombadil.sample import softplus_inverse
from tombombadil.__main__ import count_codons

class TestEstimatePiFromCounts(unittest.TestCase):
    def test_estimated_pi_sums_to_one(self):
        X = np.zeros((61, 2), dtype=int)
        X[0, 0] = 3
        X[1, 0] = 1
        X[2, 1] = 2

        pi = estimate_pi_from_counts(X, pseudocount=0.5)

        self.assertEqual((61,), pi.shape)
        self.assertAlmostEqual(1.0, pi.sum(), places=12)
        self.assertTrue(np.all(pi > 0))
        self.assertGreater(pi[0], pi[1])
        self.assertGreater(pi[1], pi[3])

    def test_zero_count_codons_get_pseudocount_probability(self):
        X = np.zeros((61, 1), dtype=int)
        X[0, 0] = 10

        pi = estimate_pi_from_counts(X, pseudocount=0.5)

        self.assertTrue(np.all(pi > 0))
        self.assertGreater(pi[0], pi[1])

    def test_zero_pseudocount_rejects_zero_probabilities(self):
        X = np.zeros((61, 1), dtype=int)
        X[0, 0] = 10

        with self.assertRaises(ValueError):
            estimate_pi_from_counts(X, pseudocount=0)

    def test_negative_pseudocount_rejected(self):
        X = np.ones((61, 1), dtype=int)

        with self.assertRaises(ValueError):
            estimate_pi_from_counts(X, pseudocount=-0.1)


class TestPiOptions(unittest.TestCase):
    def test_empirical_pi_options_parse(self):
        argv = [
            "tombombadil",
            "--alignment", "alignment.fasta",
            "--pi", "empirical",
            "--pi-pseudocount", "1.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            options = get_options()

        self.assertEqual("empirical", options.pi)
        self.assertEqual(1.25, options.pi_pseudocount)

    def test_invalid_pi_option_rejected(self):
        argv = ["tombombadil", "--alignment", "alignment.fasta", "--pi", "bad"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                get_options()


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


class TestCliDefaults(unittest.TestCase):
    def test_fitting_defaults_are_stan_unconstrained_with_eta(self):
        argv = ["tombombadil", "--alignment", "porB3.carriage.noindels.txt"]
        with mock.patch.object(sys, "argv", argv):
            args = get_options()

        self.assertEqual(args.objective_aggregate, "sum")
        self.assertEqual(args.prior_mode, "stan_unconstrained")
        self.assertFalse(args.fix_eta)
        self.assertFalse(args.fit_until_convergence)
        self.assertEqual(args.convergence_tol, 1e-6)
        self.assertEqual(args.convergence_patience, 5)
        self.assertEqual(args.convergence_check_every, 10)
        self.assertEqual(args.convergence_min_steps, 50)
        self.assertEqual(args.fit_method, "map")
        self.assertEqual(args.num_warmup, 1000)
        self.assertEqual(args.num_samples, 1000)
        self.assertEqual(args.num_chains, 4)
        self.assertEqual(args.rng_seed, 0)
        self.assertEqual(args.target_acceptance_rate, 0.8)
        self.assertEqual(args.nuts_chain_mode, "sequential")

    def test_fix_eta_flag_disables_eta_estimation(self):
        argv = ["tombombadil", "--alignment", "porB3.carriage.noindels.txt", "--fix-eta"]
        with mock.patch.object(sys, "argv", argv):
            args = get_options()

        self.assertTrue(args.fix_eta)

    def test_convergence_flags_parse(self):
        argv = [
            "tombombadil",
            "--alignment",
            "porB3.carriage.noindels.txt",
            "--fit-until-convergence",
            "--convergence-tol",
            "0.001",
            "--convergence-patience",
            "3",
            "--convergence-check-every",
            "2",
            "--convergence-min-steps",
            "4",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = get_options()

        self.assertTrue(args.fit_until_convergence)
        self.assertEqual(args.convergence_tol, 0.001)
        self.assertEqual(args.convergence_patience, 3)
        self.assertEqual(args.convergence_check_every, 2)
        self.assertEqual(args.convergence_min_steps, 4)

    def test_nuts_flags_parse(self):
        argv = [
            "tombombadil",
            "--alignment",
            "porB3.carriage.noindels.txt",
            "--fit-method",
            "nuts",
            "--num-warmup",
            "11",
            "--num-samples",
            "12",
            "--num-chains",
            "2",
            "--rng-seed",
            "9",
            "--target-acceptance-rate",
            "0.9",
            "--nuts-chain-mode",
            "pmap",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = get_options()

        self.assertEqual(args.fit_method, "nuts")
        self.assertEqual(args.num_warmup, 11)
        self.assertEqual(args.num_samples, 12)
        self.assertEqual(args.num_chains, 2)
        self.assertEqual(args.rng_seed, 9)
        self.assertEqual(args.target_acceptance_rate, 0.9)
        self.assertEqual(args.nuts_chain_mode, "pmap")

    def test_cpu_pmap_configures_jax_host_devices(self):
        argv = [
            "tombombadil",
            "--alignment",
            "porB3.carriage.noindels.txt",
            "--fit-method",
            "nuts",
            "--nuts-chain-mode",
            "pmap",
            "--cpus",
            "3",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = get_options()

        with mock.patch.dict(os.environ, {}, clear=True):
            configure_jax_for_options(args)
            self.assertEqual(
                os.environ["XLA_FLAGS"],
                "--xla_force_host_platform_device_count=3",
            )


class TestOptimizerConvergence(unittest.TestCase):
    def test_convergence_stops_before_max_steps(self):
        params = {"x": jnp.array(0.0, dtype=jnp.float64)}
        solver = optax.sgd(0.0)
        fn = lambda p: p["x"] * 0.0

        result = _optimize_params(
            fn,
            params,
            solver,
            n_iter=20,
            verbose=False,
            convergence={
                "enabled": True,
                "tol": 0.0,
                "patience": 2,
                "check_every": 1,
                "min_steps": 2,
            },
        )

        self.assertTrue(result["converged"])
        self.assertLess(result["n_steps"], 20)

    def test_fixed_step_mode_runs_requested_steps(self):
        params = {"x": jnp.array(0.0, dtype=jnp.float64)}
        solver = optax.sgd(0.0)
        fn = lambda p: p["x"] * 0.0

        result = _optimize_params(fn, params, solver, n_iter=5, verbose=False)

        self.assertFalse(result["converged"])
        self.assertEqual(result["n_steps"], 5)


class TestBlackjaxPosterior(unittest.TestCase):
    def test_posterior_summary_contains_diagnostics(self):
        raw_samples = {
            "alpha": jnp.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]], dtype=jnp.float64),
            "omega": jnp.array([[-1.0, -0.9, -0.8], [-0.9, -0.8, -0.7]], dtype=jnp.float64),
        }
        infos = {
            "acceptance_rate": jnp.array([[0.8, 0.9, 1.0], [0.7, 0.8, 0.9]]),
            "is_divergent": jnp.array([[False, False, True], [False, False, False]]),
        }

        samples, summaries, diagnostics = summarize_posterior_samples(raw_samples, infos)

        self.assertIn("alpha", samples)
        self.assertIn("alpha", summaries)
        self.assertIn("ess", summaries["alpha"])
        self.assertIn("rhat", summaries["alpha"])
        self.assertAlmostEqual(diagnostics["mean_acceptance_rate"], 0.85, places=6)
        self.assertEqual(diagnostics["n_divergent"], 1)

    def test_save_posterior_outputs_writes_samples_and_summary(self):
        raw_samples = {
            "alpha": jnp.array([[0.0, 0.1], [0.2, 0.3]], dtype=jnp.float64),
            "omega": jnp.array([[-1.0, -0.9], [-0.8, -0.7]], dtype=jnp.float64),
        }
        infos = {
            "acceptance_rate": jnp.ones((2, 2)),
            "is_divergent": jnp.zeros((2, 2), dtype=bool),
        }
        _, summaries, _ = summarize_posterior_samples(raw_samples, infos)

        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "fit")
            save_posterior_outputs(stem, raw_samples, summaries)

            self.assertTrue(os.path.exists(stem + "_posterior_samples.csv"))
            self.assertTrue(os.path.exists(stem + "_posterior_summary.csv"))
            with open(stem + "_posterior_summary.csv", newline="") as f:
                rows = {row["variable"]: row for row in csv.DictReader(f)}

        self.assertIn("alpha", rows)
        self.assertIn("omega", rows)

    def test_run_nuts_sampler_shapes_on_tiny_density(self):
        fn = lambda p: -0.5 * jnp.square(p["alpha"])
        start = {"alpha": jnp.array(0.1, dtype=jnp.float64)}

        result = run_nuts_sampler(
            fn,
            start,
            num_warmup=5,
            num_samples=6,
            num_chains=2,
            rng_seed=123,
            target_acceptance_rate=0.8,
            print_summary=False,
        )

        self.assertEqual(result["samples"]["alpha"].shape, (2, 6))
        self.assertIn("alpha", result["summaries"])
        self.assertIn("mean_acceptance_rate", result["diagnostics"])

    def test_run_nuts_sampler_pmap_requires_enough_devices(self):
        if jax.local_device_count() >= 2:
            self.skipTest("pmap guard only applies when JAX sees fewer than two devices")

        fn = lambda p: -0.5 * jnp.square(p["alpha"])
        start = {"alpha": jnp.array(0.1, dtype=jnp.float64)}

        with self.assertRaisesRegex(ValueError, "JAX sees only"):
            run_nuts_sampler(
                fn,
                start,
                num_warmup=5,
                num_samples=6,
                num_chains=2,
                rng_seed=123,
                target_acceptance_rate=0.8,
                print_summary=False,
                chain_mode="pmap",
            )

    def test_run_nuts_sampler_pmap_shapes_on_single_chain(self):
        fn = lambda p: -0.5 * jnp.square(p["alpha"])
        start = {"alpha": jnp.array(0.1, dtype=jnp.float64)}

        result = run_nuts_sampler(
            fn,
            start,
            num_warmup=5,
            num_samples=6,
            num_chains=1,
            rng_seed=123,
            target_acceptance_rate=0.8,
            print_summary=False,
            chain_mode="pmap",
        )

        self.assertEqual(result["samples"]["alpha"].shape, (1, 6))
        self.assertIn("alpha", result["summaries"])


if __name__ == '__main__':
    unittest.main()
