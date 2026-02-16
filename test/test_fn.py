import unittest # for performing unit tests
import numpy as np
import jax.numpy as jnp

import tombombadil
from tombombadil.sample import make_fn
from tombombadil.sample import transforms
from tombombadil.sample import softplus_inverse

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
            self.assertAlmostEqual(fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1), "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(0.5), dtype=jnp.float32), jnp.size(X, axis=1))}), jnp.array(-10.213031, dtype=jnp.float32), places=3)

if __name__ == '__main__':
    unittest.main()