from .base import BaseLogger
from .cli import CLILogger
from .devnull import DevNullLogger

__all__ = [
    "BaseLogger",
    "CLILogger",
    "DevNullLogger",
]


def logger() -> BaseLogger:
    """
    Returns
    -------
    ``BaseLogger``
        Current active logger.
    """
    if BaseLogger.current_logger is None:
        BaseLogger.current_logger = CLILogger()
    return BaseLogger.current_logger
