import os
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gpu_available():
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


class TestDeviceConfiguration(unittest.TestCase):
    def _run_clean_process(self, code):
        env = os.environ.copy()
        env.pop("JAX_PLATFORMS", None)
        env.pop("XLA_FLAGS", None)
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_cpu_configuration_happens_before_jax_import(self):
        result = self._run_clean_process(
            "from tombombadil.device import configure_platform; "
            "configure_platform('cpu', cpus=3, force_cpu_devices=True); "
            "import os; print(os.environ['JAX_PLATFORMS']); "
            "print(os.environ['XLA_FLAGS'])"
        )
        self.assertEqual(result.stdout.splitlines(), [
            "cpu",
            "--xla_force_host_platform_device_count=3",
        ])

    def test_gpu_configuration_does_not_force_cpu_devices(self):
        result = self._run_clean_process(
            "from tombombadil.device import configure_platform; "
            "configure_platform('gpu', cpus=8, force_cpu_devices=True); "
            "import os; print(os.environ['JAX_PLATFORMS']); "
            "print('XLA_FLAGS' in os.environ)"
        )
        self.assertEqual(result.stdout.splitlines(), ["gpu", "False"])

    def test_platform_validation_reports_visible_backend(self):
        from tombombadil.device import validate_platform

        devices, backend = validate_platform("cpu")
        self.assertTrue(devices)
        self.assertEqual(backend, "cpu")

    @unittest.skipUnless(_gpu_available(), "GPU backend is not available")
    def test_gpu_backend_is_visible_when_installed(self):
        from tombombadil.device import validate_platform

        devices, backend = validate_platform("gpu")
        self.assertTrue(devices)
        self.assertEqual(backend, "gpu")

    @unittest.skipUnless(_gpu_available(), "GPU backend is not available")
    def test_model_smoke_runs_on_gpu(self):
        from tombombadil.sample import make_fn, softplus_inverse, transforms

        X = np.zeros((61, 1), dtype=np.int32)
        X[15, 0] = 4
        X[47, 0] = 19
        pi = np.full(61, 1 / 61)
        log_pi, pimat, pimatinv, pimult = transforms(X, pi)
        fn = make_fn(pi, log_pi, pimat, pimatinv, pimult, X, jnp.ones(1), prior_mode="none")
        params = {
            name: softplus_inverse(value)
            for name, value in {
                "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
                "delta": 1.0, "epsilon": 1.0, "eta": 1.0,
                "theta": 0.5,
            }.items()
        }
        params["omega"] = jnp.repeat(jnp.array(softplus_inverse(0.5)), 1)
        result = fn(params)
        result.block_until_ready()
        self.assertTrue(bool(jnp.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
