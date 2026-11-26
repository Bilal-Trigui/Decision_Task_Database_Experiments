"""Google Colab backend. Default for real runs (free-tier T4 for 0.6B and 1.7B).

Detects Colab, mounts Drive when compute.mount_drive is true, and reads the
Hugging Face token from HF_TOKEN. Reads compute.device, compute.mount_drive,
compute.drive_mount_point.
"""
from .base import Backend


class ColabBackend(Backend):
    name = "colab"
    default_device = "cuda"
    required_env = ("HF_TOKEN",)

    @staticmethod
    def running_in_colab():
        try:
            import google.colab  # noqa: F401

            return True
        except ImportError:
            return False

    def setup(self):
        if not self.running_in_colab():
            print("warning: compute.backend is 'colab' but this is not a Colab runtime")
        if self.compute.get("mount_drive"):
            from google.colab import drive

            drive.mount(self.compute.get("drive_mount_point", "/content/drive"))
        return super().setup()


def build(cfg):
    return ColabBackend(cfg)
