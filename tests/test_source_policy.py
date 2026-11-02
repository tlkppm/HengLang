from __future__ import annotations

import ast
import io
from pathlib import Path
import re
import tokenize
import unittest

from henglang.__main__ import ROOT


class SourcePolicyTests(unittest.TestCase):
    def sources(self):
        for directory in ["henglang", "runtime", "heng", "tools", "tests", "benchmarks", "examples"]:
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and path.suffix in {".py", ".c", ".h", ".heng"}:
                    yield path
        yield ROOT / "heng.toml"

    def test_sources_are_utf8_without_bom_and_consistent_line_endings(self):
        for path in self.sources():
            with self.subTest(path=path.relative_to(ROOT)):
                content = path.read_bytes()
                self.assertFalse(content.startswith(b"\xef\xbb\xbf"))
                normalized = content.replace(b"\r\n", b"\n")
                self.assertFalse(b"\r" in normalized, "存在独立 CR 换行")
                self.assertFalse(b"\r\n" in content and b"\n" in content.replace(b"\r\n", b""), "存在混合换行")
                content.decode("utf-8", "strict")

    def test_no_added_comments_or_docstrings(self):
        literals_and_comments = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/')
        for path in self.sources():
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                if path.suffix == ".py":
                    comments = [token for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT]
                    self.assertEqual(comments, [])
                    for node in ast.walk(ast.parse(source)):
                        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.assertIsNone(ast.get_docstring(node))
                elif path.suffix in {".c", ".h", ".heng"}:
                    comments = [match.group() for match in literals_and_comments.finditer(source) if match.group().startswith(("//", "/*"))]
                    self.assertEqual(comments, [])


if __name__ == "__main__":
    unittest.main()
