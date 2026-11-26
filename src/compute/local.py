"""Local backend: the machine the notebook is opened on, CPU by default.

Used for the smoke test. Needs no credentials: the Qwen3 models are public.
Reads compute.device.
"""
from .base import Backend


class LocalBackend(Backend):
    name = "local"
    default_device = "cpu"
    required_env = ()


def build(cfg):
    return LocalBackend(cfg)
