from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "noesis-engine"
PACKAGE_NAME = "noesis_engine"

try:
    __version__ = version(DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["DISTRIBUTION_NAME", "PACKAGE_NAME", "__version__"]
