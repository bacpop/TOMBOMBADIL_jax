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
            X[21,:] = 5
            X[22,:] = 18
            #X[15,:] = 4
            #X[47,:] = 19
            pi_test = np.array([1/61 for i in range(61)])
            log_pi, pimat, pimatinv, pimult = transforms(X, pi_test)
            mask = jnp.ones(1)

            fn = make_fn(pi_test, log_pi, pimat, pimatinv, pimult, X, mask)

            for i in range(20):
                print("i", i * 0.001)
                #print(fn({"alpha": softplus_inverse(0.08288978), "beta": softplus_inverse(0.08277921), "gamma": softplus_inverse(0.07815452), 
                #                    "delta": softplus_inverse(0.08275475), "epsilon": softplus_inverse(0.06998631), "eta": softplus_inverse(3.1111848), 
                #                    "theta": softplus_inverse(2.203008), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))}))
                #print(fn({"alpha": softplus_inverse(0.8288978), "beta": softplus_inverse(0.8277921), "gamma": softplus_inverse(0.7815452), 
                #                    "delta": softplus_inverse(0.8275475), "epsilon": softplus_inverse(0.6998631), "eta": softplus_inverse(3.1111848), 
                #                    "theta": softplus_inverse(2.203008), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))}))
                print(fn({"alpha": softplus_inverse(1), "beta": softplus_inverse(1), "gamma": softplus_inverse(1), 
                                    "delta": softplus_inverse(1), "epsilon": softplus_inverse(1), "eta": softplus_inverse(1), 
                                    "theta": softplus_inverse(0.5), "omega": jnp.repeat(jnp.array(softplus_inverse(i * 0.001), dtype=jnp.float32), jnp.size(X, axis=1))})) # param set which is like Ny98 and shows nans (replicated in Stan with omega=0.003)

if __name__ == '__main__':
    unittest.main()