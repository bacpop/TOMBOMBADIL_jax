import unittest # for performing unit tests
import numpy as np
import jax.numpy as jnp

import tombombadil
from tombombadil.sample import make_fn
from tombombadil.sample import transforms

# a test for calculating the likelihood (fn) for one codon, correct value from Stan implementation
# run via python -m unittest -v test.test_fn.Testdiv
class Testdiv(unittest.TestCase):
        def testdiv(self):
            X = np.zeros((61,1))
            X[15,:] = 4
            X[47,:] = 19
            pi_test = np.array([1/61 for i in range(61)])
            log_pi, pimat, pimatinv, pimult = transforms(X, pi_test)

            fn = make_fn(pi_test, log_pi, pimat, pimatinv, pimult, X)
            self.assertAlmostEqual(fn({"alpha": 1, "beta": 1, "gamma": 1, "delta": 1, "epsilon": 1, "eta": 1, "theta": 0.5, "omega": jnp.repeat(jnp.array(0.5, dtype=jnp.float32), jnp.size(X, axis=1))}), jnp.array(-10.213031, dtype=jnp.float32), places=3)

if __name__ == '__main__':
    unittest.main()