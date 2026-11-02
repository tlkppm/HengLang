from __future__ import annotations

import unittest
from unittest.mock import patch
import urllib.request

from tools.toolchain import PROXY, RequiredProxyHandler


class ProxyTests(unittest.TestCase):
    def test_https_download_uses_explicit_connect_proxy_even_when_bypass_is_set(self):
        request = urllib.request.Request("https://ziglang.org/download/index.json")
        handler = RequiredProxyHandler({"https": PROXY})
        with patch("urllib.request.proxy_bypass", return_value=True):
            handler.proxy_open(request, PROXY, "https")
        self.assertEqual(request.host, "127.0.0.1:10808")
        self.assertEqual(request._tunnel_host, "ziglang.org")
        self.assertEqual(request.type, "https")

    def test_invalid_proxy_is_rejected_without_direct_fallback(self):
        for proxy in ["direct", "file:///tmp/proxy", "https://127.0.0.1:10808", "http://user:secret@127.0.0.1:10808", "http://127.0.0.1"]:
            request = urllib.request.Request("https://ziglang.org/download/index.json")
            with self.subTest(proxy=proxy), self.assertRaises(RuntimeError):
                RequiredProxyHandler({}).proxy_open(request, proxy, "https")


if __name__ == "__main__":
    unittest.main()
