"""Log records reach the run's log.txt through the stdout tee."""
import logging

import pytest

from src.utils import StdoutTee, setup_logging


def test_records_reach_a_tee_installed_after_setup(tmp_path):
    # setup() configures logging before _start_run() installs the tee, as here
    setup_logging("info")
    log = logging.getLogger("src.test")
    tee = StdoutTee(file_name=str(tmp_path / "log.txt"), file_mode="w")
    try:
        log.info("purified batch %d", 3)
        log.debug("hidden at info level")
        print("third-party output")
    finally:
        tee.close()

    text = (tmp_path / "log.txt").read_text()
    assert "INFO src.test: purified batch 3" in text
    assert "hidden at info level" not in text
    assert "third-party output" in text


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown log level"):
        setup_logging("chatty")
