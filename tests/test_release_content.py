from __future__ import annotations

from hashlib import sha256
import re
import unittest

from henglang.__main__ import ROOT, execute
from henglang.formatter import format_source
from henglang.frontend import load_program


class ReleaseContentTests(unittest.TestCase):
    def test_apache_license_matches_official_normalized_text(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8").strip()
        self.assertEqual(sha256(text.encode("utf-8")).hexdigest(), "283ea6cc2997a1a70da0049e09adf9317bb60ca1b51279b65196b83a69e1996b")
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("Copyright 2026 tlkppm and HengLang contributors.", notice)

    def test_documentation_local_links_exist(self):
        documents = [ROOT / name for name in ["README.md", "README.zh-CN.md", "BENCHMARKS.md", "LEGAL.md", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md"]]
        for path in documents:
            with self.subTest(document=path.name):
                content = path.read_text(encoding="utf-8")
                for target in re.findall(r"\]\(([^)]+)\)", content):
                    if "://" in target or target.startswith("#"):
                        continue
                    destination = (path.parent / target.split("#", 1)[0]).resolve()
                    self.assertTrue(destination.is_relative_to(ROOT))
                    self.assertTrue(destination.is_file(), target)

    def test_release_documents_are_utf8_and_have_bilingual_guidance(self):
        for name in ["NOTICE", "LEGAL.md", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md", "BENCHMARKS.md"]:
            with self.subTest(document=name):
                content = (ROOT / name).read_bytes()
                self.assertFalse(content.startswith(b"\xef\xbb\xbf"))
                normalized = content.replace(b"\r\n", b"\n")
                self.assertNotIn(b"\r", normalized)
                self.assertFalse(b"\r\n" in content and b"\n" in content.replace(b"\r\n", b""))
                text = content.decode("utf-8", "strict")
                self.assertIn("English", text)
                self.assertIn("简体中文", text)

    def test_public_documents_omit_machine_identifiers(self):
        forbidden = re.compile(r"Intel(?:\(R\))? Core|AMD Ryzen|\bWindows (?:10|11)\b|\b(?:CPython|Python)\s+\d+\.\d+\.\d+\b|\b\d+(?:\.\d+)?\s*GHz\b|\b\d+\s+(?:physical cores|logical processors)\b|\d+\s*个(?:物理核心|逻辑处理器)|\bDESKTOP-[A-Za-z0-9-]+\b|[A-Za-z]:[/\\]Users[/\\]", re.IGNORECASE)
        for path in sorted(ROOT.glob("*.md")):
            with self.subTest(document=path.name):
                self.assertIsNone(forbidden.search(path.read_text(encoding="utf-8")), "公开文档包含本机标识")

    def test_quickstart_is_checked_formatted_and_runs(self):
        path = ROOT / "examples" / "quickstart.heng"
        program = load_program(path)
        self.assertTrue(program.checked)
        text = path.read_text(encoding="utf-8")
        self.assertEqual(format_source(text, str(path)), text)
        result = execute(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_source_distribution_ignores_local_artifacts(self):
        patterns = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        self.assertTrue({".tools/", ".build/", "__pycache__/", ".env", ".env.*", "*.pem", "*.key", ".cursor/"}.issubset(patterns))


if __name__ == "__main__":
    unittest.main()
