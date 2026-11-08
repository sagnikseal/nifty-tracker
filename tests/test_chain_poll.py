"""`chain-poll`: one option-chain snapshot for a short scheduled job (no real network; fully mocked)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

os.environ.setdefault("NE_DATA_DIR", tempfile.mkdtemp(prefix="ne_test_"))

from nifty_engine import __main__ as cli  # noqa: E402


def _run():
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(["chain-poll"])
    return json.loads(buf.getvalue())


class TestChainPoll(unittest.TestCase):
    def test_skips_without_token(self):
        with mock.patch("nifty_engine.data.providers.UpstoxProvider.available", new_callable=mock.PropertyMock,
                        return_value=False):
            self.assertEqual(_run(), {"skipped": "no Upstox token set"})

    def test_skips_when_market_closed(self):
        with mock.patch("nifty_engine.data.providers.UpstoxProvider.available", new_callable=mock.PropertyMock,
                        return_value=True), \
             mock.patch("nifty_engine.calendar.is_market_open", return_value=False):
            self.assertEqual(_run(), {"skipped": "market is closed"})

    def test_polls_once_when_open(self):
        with mock.patch("nifty_engine.data.providers.UpstoxProvider.available", new_callable=mock.PropertyMock,
                        return_value=True), \
             mock.patch("nifty_engine.calendar.is_market_open", return_value=True), \
             mock.patch("nifty_engine.data.recorder.ChainPoller.poll_once", return_value={"ts": "2026-11-09T10:00:00+05:30",
                                                                                          "near_pcr_oi": 1.1}):
            self.assertEqual(_run()["near_pcr_oi"], 1.1)


if __name__ == "__main__":
    unittest.main()
