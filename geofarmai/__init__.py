"""Import-safe public API for GeoFarmAI's scientific data model."""

from .data import (
    CoordinateCandidates,
    CoordinateSpec,
    DataSource,
    FieldDataset,
    SchemaInspection,
    VariableRole,
    VariableSpec,
    coordinate_candidates,
    inspect_schema,
)
from .exceptions import (
    DataModelError,
    DataSourceError,
    GeoFarmAIError,
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
    "RMultispatiUnavailableError",
    "SchemaInspection",
    "SchemaValidationError",
    "VariableRole",
    "VariableSpec",
    "coordinate_candidates",
    "inspect_schema",
    "__version__",
]

__version__ = "0.1.0"
