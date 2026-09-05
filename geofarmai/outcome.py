"""Outcome configuration for scientific pipeline orchestration.

Scientific roles are explicit. No column becomes an outcome because of its
name. The only yield-specific behavior in this module is the isolated legacy
configuration adapter at the bottom of the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from geofarmai.exceptions import OutcomeConfigurationError


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """Resolved optional outcome used for external cluster validation."""

    name: str
    source_column: str
    source_config: Mapping[str, Any] | None = None
    legacy_yield_adapter: bool = False

    @property
    def uses_predictor_source(self) -> bool:
        return self.source_config is None


def resolve_pipeline_outcome(project: Mapping[str, Any]) -> PipelineOutcome | None:
    """Resolve an explicit generic outcome or adapt the legacy yield section.

    New configuration accepts either a column name in the predictor source::

        project:
          outcome: nitrate

    or a separate source::

        project:
          outcome:
            name: nitrate
            column: nitrate_mg_kg
            path: data/nitrate.csv
            x: Longitude
            y: Latitude

    ``outcome: null`` explicitly selects an unsupervised analysis. Legacy
    ``project.yield`` configuration is handled only by
    ``_resolve_legacy_yield_outcome``.
    """

    if "outcome" in project:
        declaration = project["outcome"]
        if declaration is None:
            return None
        if isinstance(declaration, str):
            name = _nonempty_name(declaration, "project.outcome")
            return PipelineOutcome(name=name, source_column=name)
        if not isinstance(declaration, Mapping):
            raise OutcomeConfigurationError(
                "project.outcome must be a column name, a mapping, or null."
            )

        source_column = declaration.get("column", declaration.get("name"))
        if source_column is None:
            raise OutcomeConfigurationError(
                "A mapped project.outcome requires 'column' or 'name'."
            )
        source_column = _nonempty_name(source_column, "project.outcome.column")
        name = _nonempty_name(declaration.get("name", source_column), "project.outcome.name")
        source_config = None
        if "path" in declaration:
            missing = [key for key in ("path", "x", "y") if not declaration.get(key)]
            if missing:
                raise OutcomeConfigurationError(
                    f"External project.outcome is missing required fields: {missing}."
                )
            source_config = dict(declaration)
        return PipelineOutcome(
            name=name,
            source_column=source_column,
            source_config=source_config,
        )

    return _resolve_legacy_yield_outcome(project)


def _resolve_legacy_yield_outcome(
    project: Mapping[str, Any],
) -> PipelineOutcome | None:
    """Translate the historical yield configuration into a generic outcome."""

    if "yield" not in project:
        if "yield_column" in project:
            internal_name = _nonempty_name(
                project["yield_column"], "project.yield_column"
            )
            return PipelineOutcome(
                name=internal_name,
                source_column=internal_name,
                legacy_yield_adapter=True,
            )
        return None
    declaration = project["yield"]
    if not isinstance(declaration, Mapping):
        raise OutcomeConfigurationError("Legacy project.yield must be a mapping.")
    missing = [key for key in ("path", "x", "y", "column") if not declaration.get(key)]
    if missing:
        raise OutcomeConfigurationError(
            f"Legacy project.yield is missing required fields: {missing}."
        )
    source_column = _nonempty_name(declaration["column"], "project.yield.column")
    internal_name = _nonempty_name(
        project.get("yield_column", "yield"), "project.yield_column"
    )
    return PipelineOutcome(
        name=internal_name,
        source_column=source_column,
        source_config=dict(declaration),
        legacy_yield_adapter=True,
    )


def _nonempty_name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutcomeConfigurationError(f"{field} must be a non-empty string.")
    return value.strip()
