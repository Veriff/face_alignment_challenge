import logging
import logging.handlers
from datetime import datetime

LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class IsoDateFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).isoformat()


def create_formatter(fmt: str = None):
    if fmt is None:
        fmt = LOGGING_FORMAT

    return IsoDateFormatter(fmt=fmt)


def setup_logging(*, filename: str = None, debug: bool = False):
    handler = logging.StreamHandler() if filename is None else logging.handlers.WatchedFileHandler(filename)
    handler.setFormatter(create_formatter())

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, handlers=[handler])
