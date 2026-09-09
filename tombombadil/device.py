"""JAX backend configuration performed before importing JAX."""

import os


SUPPORTED_PLATFORMS = ("cpu", "gpu", "tpu")


def configure_platform(platform, *, cpus=1, force_cpu_devices=False):
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")
    os.environ["JAX_PLATFORMS"] = platform
    if force_cpu_devices and platform == "cpu":
        flag = f"--xla_force_host_platform_device_count={max(int(cpus), 1)}"
        existing = os.environ.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count" not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


def validate_platform(platform):
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")
    import jax
    devices = list(jax.devices(platform))
    if not devices:
        raise RuntimeError(f"No JAX devices are available for platform {platform!r}")
    return devices, jax.default_backend()
