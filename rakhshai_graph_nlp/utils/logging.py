import logging
from typing import Union

_LEVEL = Union[int, str]

def setup_logger(level: _LEVEL = "INFO") -> logging.Logger:
    """Return a configured ``rgnn`` logger.

    Parameters
    ----------
    level:
        Logging level or its string representation.
    """

    logger = logging.getLogger("rgnn")
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
