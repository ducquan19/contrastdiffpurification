# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Helpers shared by the entry points: logging setup, a stdout/stderr tee and a bool CLI type."""
import argparse
import logging
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _CurrentStdoutHandler(logging.StreamHandler):
    """StreamHandler that always writes to the *current* ``sys.stdout``.

    A plain StreamHandler keeps the stream it was created with, so records would bypass a
    ``StdoutTee`` installed later and never reach the run's log file.
    """

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def setup_logging(level: str = "info") -> None:
    """Send records of every logger to stdout with a timestamp; ``level`` is a logging level name."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level '{level}'")
    handler = _CurrentStdoutHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(numeric)


class StdoutTee(object):
    """
    Mirror stdout and stderr (log records, third-party prints, tracebacks) into a file,
    optionally flushing after every write.
    """

    def __init__(self, file_name: Optional[str] = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if several tees are stacked, only undo our own redirection
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def str2bool(v):
    """argparse ``type`` that accepts yes/no, true/false, t/f, y/n, 1/0 (case-insensitive)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
