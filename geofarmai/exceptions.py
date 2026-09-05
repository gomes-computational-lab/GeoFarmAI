"""GeoFarmAI-specific exceptions."""


class GeoFarmAIError(Exception):
    """Base exception for errors reported by GeoFarmAI."""


class DataModelError(GeoFarmAIError, ValueError):
    """Base exception for invalid canonical data-model inputs."""


class SchemaValidationError(DataModelError):
    """Raised when variable or coordinate declarations are invalid."""


class DataSourceError(DataModelError):
    """Raised when an input source cannot form a valid canonical dataset."""


class HarmonizationError(GeoFarmAIError, ValueError):
    """Raised when spatial sources cannot be harmonized safely."""


class RMultispatiUnavailableError(GeoFarmAIError, RuntimeError):
    """Raised when explicitly requested R MULTISPATI support cannot run."""
