class CanonError(Exception):
    """Base class for all Canon errors."""


class ConfigError(CanonError):
    """Invalid or missing configuration."""


class JudgeError(CanonError):
    """The judge failed to produce a usable answer."""


class RubricError(CanonError):
    """Invalid rubric definition."""
