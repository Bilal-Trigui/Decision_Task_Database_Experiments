"""Azure backend: an Azure GPU VM or ML workspace, for bigger models.

Reads the subscription, resource group and workspace from environment
variables and prints `LOGIN CREDENTIALS NEEDED HERE: <VAR>` for each one that
is missing. Reads compute.device.
"""
from .base import Backend


class AzureBackend(Backend):
    name = "azure"
    default_device = "cuda"
    required_env = (
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_ML_WORKSPACE",
        "HF_TOKEN",
    )


def build(cfg):
    return AzureBackend(cfg)
