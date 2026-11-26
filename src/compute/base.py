"""Base class for compute backends.

Every backend reads only the `compute` block of the settings and environment
variables. Credentials never live in the settings file: a backend names the
environment variables it needs in `required_env`, and `credentials()` prints
the literal line `LOGIN CREDENTIALS NEEDED HERE: <VAR>` for each one that is
missing and stops.
"""
import os
import sys

CREDENTIALS_MESSAGE = "LOGIN CREDENTIALS NEEDED HERE"
WINDOWS_GPU_MEMORY_FRACTION = 0.9


class MissingCredentials(RuntimeError):
    """Raised after the missing-variable lines have been printed."""


class Backend:
    name = "base"
    default_device = "cuda"
    required_env = ()

    def __init__(self, cfg):
        self.cfg = cfg
        self.compute = cfg["compute"]

    def credentials(self):
        """Read every variable in `required_env` from the environment. Reads compute."""
        found, missing = {}, []
        for var in self.required_env:
            value = os.environ.get(var)
            if value:
                found[var] = value
            else:
                missing.append(var)
        for var in missing:
            print(f"{CREDENTIALS_MESSAGE}: {var}")
        if missing:
            raise MissingCredentials(
                f"backend '{self.name}' needs environment variables: {', '.join(missing)}"
            )
        return found

    def get_device(self):
        """Return 'cpu' or 'cuda', checking that a requested GPU exists. Reads compute.device."""
        device = self.compute.get("device") or self.default_device
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"compute.device is 'cuda' but no CUDA device is visible on backend '{self.name}'."
                )
            if sys.platform == "win32":
                # On Linux a full card makes the allocation fail, and PyTorch then frees its
                # cached blocks and retries. On Windows the driver lends system RAM instead
                # (sysmem fallback), so nothing ever fails, the cache is never freed, and the
                # run grows into RAM. A per-process cap restores the Linux behaviour.
                torch.cuda.set_per_process_memory_fraction(WINDOWS_GPU_MEMORY_FRACTION)
        return device

    def setup(self):
        """Check credentials and prepare the environment. Reads compute."""
        creds = self.credentials()
        if "HF_TOKEN" in creds:
            os.environ["HF_TOKEN"] = creds["HF_TOKEN"]
        print(f"compute backend: {self.name}, device: {self.get_device()}")
        return creds
