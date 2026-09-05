"""Canonical, role-explicit GeoFarmAI data model."""

from geofarmai.data.dataset import FieldDataset
from geofarmai.data.inspect import (
    CoordinateCandidates,
    SchemaInspection,
    coordinate_candidates,
    inspect_schema,
)
from geofarmai.data.harmonization import (
    HarmonizationWarning,
    HarmonizedFieldDataset,
    VariableIdentity,
    VariableProvenance,
    harmonize,
)
from geofarmai.data.schema import CoordinateSpec, VariableRole, VariableSpec
from geofarmai.data.source import DataSource

__all__ = [
    "CoordinateCandidates",
    "CoordinateSpec",
    "DataSource",
    "FieldDataset",
    "HarmonizationWarning",
    "HarmonizedFieldDataset",
    "SchemaInspection",
    "VariableRole",
    "VariableIdentity",
    "VariableProvenance",
    "VariableSpec",
    "coordinate_candidates",
    "harmonize",
    "inspect_schema",
]
