"""Import-safe public API for GeoFarmAI's scientific data model."""

from .data import (
    CoordinateCandidates,
    CoordinateSpec,
    DataSource,
    FieldDataset,
    HarmonizationWarning,
    HarmonizedFieldDataset,
    SchemaInspection,
    VariableRole,
    VariableIdentity,
    VariableProvenance,
    VariableSpec,
    coordinate_candidates,
    harmonize,
    inspect_schema,
)
from .exceptions import (
    DataModelError,
    DataSourceError,
    GeoFarmAIError,
    HarmonizationError,
    ModelConfigurationError,
    ModelNotFittedError,
    MultispatiUnavailableError,
    OutcomeConfigurationError,
    RMultispatiUnavailableError,
    SchemaValidationError,
)
__all__ = [
    "CoordinateCandidates",
    "CoordinateSpec",
    "CandidateSolution",
    "DataModelError",
    "DataSource",
    "DataSourceError",
    "FieldDataset",
    "GeoFarmAIError",
    "GeoFarmModel",
    "GeoFarmResult",
    "HarmonizationError",
    "HarmonizationWarning",
    "HarmonizedFieldDataset",
    "ModelConfigurationError",
    "ModelNotFittedError",
    "MultispatiUnavailableError",
    "OutcomeConfigurationError",
    "RMultispatiUnavailableError",
    "SchemaInspection",
    "SchemaValidationError",
    "VariableRole",
    "VariableIdentity",
    "VariableProvenance",
    "VariableSpec",
    "coordinate_candidates",
    "harmonize",
    "inspect_schema",
    "run_pipeline",
    "__version__",
]

__version__ = "0.1.0"


def __getattr__(name):
    """Load model orchestration lazily and keep base imports side-effect free."""

    if name == "GeoFarmModel":
        from .model import GeoFarmModel

        return GeoFarmModel
    if name in {"CandidateSolution", "GeoFarmResult"}:
        from .result import CandidateSolution, GeoFarmResult

        return {"CandidateSolution": CandidateSolution, "GeoFarmResult": GeoFarmResult}[name]
    if name == "run_pipeline":
        from .api import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
