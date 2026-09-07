"""Hosted fine-tuning API backend: a stub.

This is how Plunkett et al. ran their experiment (OpenAI's fine-tuning API).
It is kept as a backend so that path stays expressible in the settings file,
but it is not implemented. Reads compute.
"""
from .base import Backend


class ApiBackend(Backend):
    name = "api"
    default_device = "cpu"
    required_env = ()

    def setup(self):
        raise NotImplementedError(
            "compute.backend 'api' is where a hosted fine-tuning API (Plunkett's OpenAI "
            "path) would plug in: upload the JSONL built by src/data.py, create a job, poll "
            "it, and query the resulting model. Not implemented; use local, colab or azure."
        )


def build(cfg):
    return ApiBackend(cfg)
