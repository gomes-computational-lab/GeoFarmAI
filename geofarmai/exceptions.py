"""GeoFarmAI-specific exceptions."""


class GeoFarmAIError(Exception):
    """Base exception for errors reported by GeoFarmAI."""


class RMultispatiUnavailableError(GeoFarmAIError, RuntimeError):
    """Raised when explicitly requested R MULTISPATI support cannot run."""
