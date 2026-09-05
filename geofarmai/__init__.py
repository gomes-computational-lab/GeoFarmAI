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
    RMultispatiUnavailableError,
    SchemaValidationError,
)

__all__ = [
    "CoordinateCandidates",
    "CoordinateSpec",
    "DataModelError",
    "DataSource",
    "DataSourceError",
    "FieldDataset",
    "GeoFarmAIError",
    "HarmonizationError",
    "HarmonizationWarning",
    "HarmonizedFieldDataset",
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
    "__version__",
]

__version__ = "0.1.0"
