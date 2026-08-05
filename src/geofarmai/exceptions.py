"""GeoFarmAI exception hierarchy."""


class GeoFarmAIError(Exception):
    """Base class for expected GeoFarmAI failures."""


class ConfigurationError(GeoFarmAIError, ValueError):
    """Raised when configuration is missing, inconsistent, or unsupported."""


class InputDataError(GeoFarmAIError, ValueError):
    """Raised when an input file or required input column is invalid."""


class OptionalDependencyError(GeoFarmAIError, ImportError):
    """Raised when an explicitly selected optional pathway is unavailable."""
