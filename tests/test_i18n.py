"""English is the default UI language; every translatable string must have a catalog entry."""
import ast
import json
import string
import unittest
from pathlib import Path

from barforge import i18n
from barforge.i18n import english_catalog, set_language, tr

PACKAGE = Path(i18n.__file__).parent
CYRILLIC = {chr(code) for code in range(0x0400, 0x0500)}
# Russian text that is data rather than UI: it parses what the Grok model answers.
DATA_LITERALS = {r"(?i)^(caption|капшен|подпись)\s*:\s*"}


def _modules():
    return sorted(path for path in PACKAGE.glob("*.py") if path.name != "i18n.py")


def _is_translated_call(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tr"


def _placeholders(text):
    return sorted(field for _, field, _, _ in string.Formatter().parse(text) if field)


class LocalizationCoverageTests(unittest.TestCase):
    def setUp(self):
        set_language("en")

    def tearDown(self):
        set_language("en")

    def test_every_translated_key_has_matching_english(self):
        catalog = english_catalog()
        problems = []
        for path in _modules():
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not (_is_translated_call(node) and node.args and isinstance(node.args[0], ast.Constant)):
                    continue
                key = node.args[0].value
                english = catalog.get(key)
                if english is None:
                    problems.append(f"{path.name}:{node.lineno}: missing English for {key!r}")
                elif node.keywords and _placeholders(key) != _placeholders(english):
                    problems.append(f"{path.name}:{node.lineno}: placeholders differ for {key!r}")
        self.assertEqual(problems, [])

    def test_no_russian_text_escapes_translation(self):
        problems = []
        for path in _modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = {id(node.value) for node in ast.walk(tree)
                          if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
            translated = {id(node.args[0]) for node in ast.walk(tree) if _is_translated_call(node) and node.args}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and CYRILLIC & set(node.value) and id(node) not in docstrings
                        and id(node) not in translated and node.value not in DATA_LITERALS):
                    problems.append(f"{path.name}:{node.lineno}: {node.value[:50]!r}")
        self.assertEqual(problems, [])

    def test_catalog_has_no_unused_or_russian_values(self):
        used = set()
        for path in _modules():
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if _is_translated_call(node) and node.args and isinstance(node.args[0], ast.Constant):
                    used.add(node.args[0].value)
        catalog = english_catalog()
        self.assertEqual(sorted(set(catalog) - used), [])
        self.assertEqual([key for key, value in catalog.items() if CYRILLIC & set(value)], [])

    def test_catalog_files_are_valid_flat_json(self):
        for path in (PACKAGE / "locales").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()), path.name)

    def test_language_selects_text_and_formats_placeholders_after_translation(self):
        key = "Удалено полос: {value}"
        self.assertEqual(tr(key, value=3), "Bars removed: 3")
        set_language("ru")
        self.assertEqual(tr(key, value=3), "Удалено полос: 3")
        self.assertEqual(tr("Нет такой строки"), "Нет такой строки")

    def test_unknown_language_is_rejected(self):
        with self.assertRaises(ValueError):
            set_language("de")


if __name__ == "__main__":
    unittest.main()
