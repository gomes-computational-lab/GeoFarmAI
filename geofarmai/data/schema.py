"""Variable and coordinate declarations for GeoFarmAI input data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyproj import CRS
from pyproj.exceptions import CRSError

from geofarmai.exceptions import SchemaValidationError


class VariableRole(str, Enum):
    """Scientific and descriptive roles available to input variables."""

    PREDICTOR = "predictor"
    OUTCOME = "outcome"
    COORDINATE = "coordinate"
    IDENTIFIER = "identifier"
    METADATA = "metadata"


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Explicitly declare how one named variable participates in an analysis.

    ``domain`` is descriptive metadata only. It never determines the role or
    changes scientific processing.
    """

    name: str
    role: VariableRole | str
    domain: str | None = None
    units: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SchemaValidationError("Variable names must be non-empty strings.")
        object.__setattr__(self, "name", self.name.strip())

        try:
            role = VariableRole(self.role)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(role.value for role in VariableRole)
            raise SchemaValidationError(
                f"Invalid role {self.role!r} for variable {self.name!r}. "
                f"Allowed roles are: {allowed}."
            ) from exc
        object.__setattr__(self, "role", role)

        for field_name in ("domain", "units", "description"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise SchemaValidationError(
                    f"{field_name.capitalize()} for variable {self.name!r} must be a string or None."
                )


@dataclass(frozen=True, slots=True)
class CoordinateSpec:
    """Declare coordinate columns and their optional coordinate reference system.

    ``x`` and ``y`` may both be omitted only for a GeoDataFrame whose active
    geometry is the coordinate representation.
    """

    x: str | None = None
    y: str | None = None
    crs: CRS | str | int | None = None

    def __post_init__(self) -> None:
        if (self.x is None) != (self.y is None):
            raise SchemaValidationError(
                "CoordinateSpec requires both x and y column names, or neither for geometry-backed coordinates."
            )

        if self.x is not None and self.y is not None:
            if not isinstance(self.x, str) or not self.x.strip():
                raise SchemaValidationError("The x coordinate column name must be a non-empty string.")
            if not isinstance(self.y, str) or not self.y.strip():
                raise SchemaValidationError("The y coordinate column name must be a non-empty string.")
            x = self.x.strip()
            y = self.y.strip()
            if x == y:
                raise SchemaValidationError("The x and y coordinate columns must be different.")
            object.__setattr__(self, "x", x)
            object.__setattr__(self, "y", y)

        if self.crs is not None:
            try:
                parsed = CRS.from_user_input(self.crs)
            except (CRSError, TypeError, ValueError) as exc:
                raise SchemaValidationError(f"Invalid CRS {self.crs!r}.") from exc
            object.__setattr__(self, "crs", parsed)

    @property
    def uses_geometry(self) -> bool:
        """Return whether coordinates come from GeoDataFrame geometry."""

        return self.x is None and self.y is None

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the declared coordinate columns, if any."""

        if self.x is None or self.y is None:
            return ()
        return self.x, self.y

    @property
    def is_geographic(self) -> bool | None:
        """Return the CRS geographic flag, or ``None`` when CRS is unknown."""

        return None if self.crs is None else bool(self.crs.is_geographic)

    @property
    def is_projected(self) -> bool | None:
        """Return the CRS projected flag, or ``None`` when CRS is unknown."""

        return None if self.crs is None else bool(self.crs.is_projected)
