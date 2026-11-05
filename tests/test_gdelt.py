"""GDELT client: throttling, 429 retries, cache and circuit breaker (no network)."""

import json
import tempfile
import time
import unittest

from nifty_engine.data.providers import GdeltError, GdeltProvider


class FakeResp:
    def __init__(self, status=200, body=None, text=None, headers=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})
        self.headers = headers or {"Content-Type": "application/json" if body is not None else "text/plain"}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((time.time(), dict(params)))
        return self.responses.pop(0)


ARTS = {"articles": [{"seendate": "20261104T020000Z", "title": "RBI holds rates", "domain": "x.com", "url": "u"}]}
THROTTLE = FakeResp(200, None, text="Please limit requests to one every 5 seconds or contact ...")


def provider(responses, sleeps):
    return GdeltProvider(session=FakeSession(responses), cache_dir=tempfile.mkdtemp(),
                         sleep=lambda s: sleeps.append(s))


class TestGdelt(unittest.TestCase):
    def test_retries_429_then_succeeds(self):
        sleeps = []
        g = provider([FakeResp(429, None, text="Too Many Requests"), THROTTLE, FakeResp(200, ARTS)], sleeps)
        import datetime as dt
        end = dt.datetime(2026, 11, 4, 9, 0, tzinfo=dt.timezone.utc)
        out = g.articles("q", end - dt.timedelta(hours=3), end)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(g.s.calls), 3)
        self.assertIn(8.0, sleeps)       # backoff after the first 429
        self.assertIn(16.0, sleeps)      # doubled after the plain-text throttle page

    def test_honours_retry_after(self):
        sleeps = []
        g = provider([FakeResp(429, None, text="slow down", headers={"Retry-After": "11"}), FakeResp(200, ARTS)], sleeps)
        import datetime as dt
        end = dt.datetime(2026, 11, 4, 9, 0, tzinfo=dt.timezone.utc)
        g.articles("q", end - dt.timedelta(hours=3), end)
        self.assertIn(11.0, sleeps)

    def test_throttles_consecutive_calls(self):
        sleeps = []
        g = provider([FakeResp(200, ARTS), FakeResp(200, ARTS)], sleeps)
        import datetime as dt
        end = dt.datetime(2026, 11, 4, 9, 0, tzinfo=dt.timezone.utc)
        g.articles("q1", end - dt.timedelta(hours=3), end)
        g.articles("q2", end - dt.timedelta(hours=3), end)
        self.assertTrue(any(4.0 < s <= GdeltProvider.MIN_INTERVAL for s in sleeps))

    def test_past_window_is_cached(self):
        sleeps = []
        g = provider([FakeResp(200, ARTS)], sleeps)
        import datetime as dt
        end = dt.datetime(2026, 9, 1, 3, 40, tzinfo=dt.timezone.utc)
        a = g.articles("q", end - dt.timedelta(hours=18), end)
        b = g.articles("q", end - dt.timedelta(hours=18), end)     # served from disk, no 2nd request
        self.assertEqual(a, b)
        self.assertEqual(len(g.s.calls), 1)

    def test_circuit_breaker(self):
        sleeps = []
        g = provider([FakeResp(429, None, text="x")] * (GdeltProvider.MAX_RETRIES + 1), sleeps)
        import datetime as dt
        end = dt.datetime(2026, 11, 4, 9, 0, tzinfo=dt.timezone.utc)
        with self.assertRaises(GdeltError):
            g.articles("q", end - dt.timedelta(hours=3), end)
        n = len(g.s.calls)
        with self.assertRaises(GdeltError):                         # cooling down: no new request
            g.articles("q2", end - dt.timedelta(hours=3), end)
        self.assertEqual(len(g.s.calls), n)


if __name__ == "__main__":
    unittest.main()
