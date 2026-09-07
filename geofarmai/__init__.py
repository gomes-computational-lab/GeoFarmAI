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
    ConfigurationError,
    DataModelError,
    DataSourceError,
    GeoFarmAIError,
    HarmonizationError,
    ModelConfigurationError,
    ModelNotFittedError,
    MultispatiUnavailableError,
    OptionalDependencyError,
    OutcomeConfigurationError,
    RMultispatiUnavailableError,
    SchemaValidationError,
)
__all__ = [
    "CoordinateCandidates",
    "CoordinateSpec",
    "CandidateSolution",
    "ConfigurationError",
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
    "OptionalDependencyError",
    "OutcomeConfigurationError",
    "RMultispatiUnavailableError",
    "SchemaInspection",
    "SchemaValidationError",
    "VariableRole",
    "VariableIdentity",
    "VariableProvenance",
    "VariableSpec",
    "coordinate_candidates",
    "delineate_zones",
    "harmonize",
    "inspect_schema",
    "run_pipeline",
    "load_config",
    "load_legacy_config",
    "run_from_config",
    "AnalysisConfiguration",
    "GeoFarmConfiguration",
    "SourceConfiguration",
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
    if name in {"delineate_zones", "run_pipeline"}:
        from .api import delineate_zones, run_pipeline

        return {
            "delineate_zones": delineate_zones,
            "run_pipeline": run_pipeline,
        }[name]
    if name in {
        "AnalysisConfiguration",
        "GeoFarmConfiguration",
        "SourceConfiguration",
        "load_config",
        "load_legacy_config",
        "run_from_config",
    }:
        from .config import (
            AnalysisConfiguration,
            GeoFarmConfiguration,
            SourceConfiguration,
            load_config,
            load_legacy_config,
            run_from_config,
        )

        return {
            "AnalysisConfiguration": AnalysisConfiguration,
            "GeoFarmConfiguration": GeoFarmConfiguration,
            "SourceConfiguration": SourceConfiguration,
            "load_config": load_config,
            "load_legacy_config": load_legacy_config,
            "run_from_config": run_from_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
