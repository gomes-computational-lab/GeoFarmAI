"""Canonical, role-explicit GeoFarmAI data model."""

from geofarmai.data.dataset import FieldDataset
from geofarmai.data.inspect import (
    CoordinateCandidates,
    SchemaInspection,
    coordinate_candidates,
    inspect_schema,
)
from geofarmai.data.schema import CoordinateSpec, VariableRole, VariableSpec
from geofarmai.data.source import DataSource

__all__ = [
    "CoordinateCandidates",
    "CoordinateSpec",
    "DataSource",
    "FieldDataset",
    "SchemaInspection",
    "VariableRole",
    "VariableSpec",
    "coordinate_candidates",
    "inspect_schema",
]
