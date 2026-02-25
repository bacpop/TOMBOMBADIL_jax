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
                print("i", i * 0.0003)
                #print(fn({"alpha": softplus_inverse(0.08288978), "beta": softplus_inverse(0.08277921), "gamma": softplus_inverse(0.07815452), 
                #                    "delta": softplus_inverse(0.08275475), "epsilon": softplus_inverse(0.06998631), "eta": softplus_inverse(3.1111848), 
                #                    "theta": softplus_inverse(2.203008), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))}))
                #print(fn({"alpha": softplus_inverse(0.8288978), "beta": softplus_inverse(0.8277921), "gamma": softplus_inverse(0.7815452), 
                #                    "delta": softplus_inverse(0.8275475), "epsilon": softplus_inverse(0.6998631), "eta": softplus_inverse(3.1111848), 
                #                    "theta": softplus_inverse(2.203008), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))}))
                print(fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), 
                                    "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1), 
                                    "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))})) # param set which is like Ny98 and shows nans (replicated in Stan with omega=0.003)
                fn_result[i] = fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), 
                                    "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1), 
                                    "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))})
                fn_i[i] = i * 0.0003
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

if __name__ == '__main__':
    unittest.main()