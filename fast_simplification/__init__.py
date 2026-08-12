from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .replay import _map_isolated_points, replay_simplification  # noqa: F401
from .simplify import simplify, simplify_mesh  # noqa: F401

try:
    __version__ = _version("fast_simplification")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
