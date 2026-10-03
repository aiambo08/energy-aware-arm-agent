"""armbench: energy-aware LLM agent benchmark for a simulated UR5e arm."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("armbench")
except PackageNotFoundError:  # pragma: no cover - editable installs without metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
