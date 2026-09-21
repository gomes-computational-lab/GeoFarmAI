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
    TerrainProcessingError,
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
    "TerrainDerivatives",
    "TerrainProcessingError",
    "TerrainResult",
    "VariableRole",
    "VariableIdentity",
    "VariableProvenance",
    "VariableSpec",
    "coordinate_candidates",
    "delineate_zones",
    "harmonize",
    "inspect_schema",
    "derive_terrain",
    "terrain_derivatives",
    "terrain_from_elevation_points",
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
    if name in {
        "TerrainDerivatives",
        "TerrainResult",
        "derive_terrain",
        "terrain_derivatives",
        "terrain_from_elevation_points",
    }:
        from .terrain import (
            TerrainDerivatives,
            TerrainResult,
            derive_terrain,
            terrain_derivatives,
            terrain_from_elevation_points,
        )

        return {
            "TerrainDerivatives": TerrainDerivatives,
            "TerrainResult": TerrainResult,
            "derive_terrain": derive_terrain,
            "terrain_derivatives": terrain_derivatives,
            "terrain_from_elevation_points": terrain_from_elevation_points,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
