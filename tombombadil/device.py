"""JAX platform selection and runtime validation.

This module deliberately does not import JAX at module import time.  The CLI
uses :func:`configure_platform` before importing the numerical modules, which
allows JAX's platform configuration to be selected through the environment.
"""

from __future__ import annotations

import logging
import os
import sys


SUPPORTED_PLATFORMS = ("cpu", "gpu")


def configure_platform(platform: str, *, cpus: int = 1, force_cpu_devices: bool = False) -> None:
    """Configure the JAX platform before JAX is imported.

    CPU remains explicit and deterministic by default.  GPU selection is
    strict: if the requested GPU backend is not installed, JAX will report a
    clear error rather than silently falling back to CPU.
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")
    if "jax" in sys.modules:
        # Keep direct-library callers safe when they request the backend that
        # JAX is already using, while rejecting an ineffective late switch.
        loaded_jax = sys.modules["jax"]
        try:
            active_backend = loaded_jax.default_backend()
        except AttributeError:
            active_backend = None
        if active_backend is not None and active_backend != platform:
            raise RuntimeError(
                "JAX has already been imported with backend "
                f"'{active_backend}'; platform selection must happen before "
                "importing the numerical modules."
            )

    os.environ["JAX_PLATFORMS"] = platform

    if force_cpu_devices and platform == "cpu":
        cpu_devices = max(int(cpus), 1)
        flag = f"--xla_force_host_platform_device_count={cpu_devices}"
        existing = os.environ.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count" not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


def validate_platform(platform: str) -> tuple[list[object], str]:
    """Return visible devices and backend name, or raise an actionable error."""
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")

    try:
        import jax

        devices = list(jax.devices(platform))
    except Exception as exc:  # JAX reports backend/plugin failures here.
        raise RuntimeError(
            f"JAX platform '{platform}' is unavailable. Install the matching "
            "JAX accelerator extra and verify the driver/runtime installation."
        ) from exc

    if not devices:
        raise RuntimeError(f"No JAX devices are available for platform '{platform}'.")

    backend = jax.default_backend()
    if backend != platform:
        logging.warning(
            "Requested JAX platform '%s', but the default backend is '%s'.",
            platform,
            backend,
        )
    return devices, backend


def platform_summary(platform: str) -> str:
    """Return a compact, user-facing description of the selected devices."""
    devices, backend = validate_platform(platform)
    device_names = ", ".join(str(device) for device in devices)
    try:
        import jax

        x64 = bool(jax.config.jax_enable_x64)
    except Exception:
        x64 = False
    return f"platform={platform}, backend={backend}, x64={x64}, devices=[{device_names}]"
