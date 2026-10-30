"""Restart-based UI translator; Russian source strings are the catalog keys.

The active language is fixed once, before any ``barforge`` UI module is imported
(``initialize_language`` in ``app.main``), so modules may call ``tr`` at import time.
A saved language change applies on the next start; a window never mixes languages.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_LANGUAGE = "en"
LANGUAGE_NAMES = {"en": "English", "ru": "Русский"}
SUPPORTED_LANGUAGES = frozenset(LANGUAGE_NAMES)
_SETTINGS_KEY = "language"
_LANGUAGE = DEFAULT_LANGUAGE


def _load_english() -> dict[str, str]:
    catalog: dict[str, str] = {}
    directory = Path(__file__).with_name("locales")
    if not directory.is_dir():
        return catalog
    for path in sorted(directory.glob("en*.json")):
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in entries.items()):
            raise RuntimeError(f"Invalid translation catalog: {path.name}")
        for key, value in entries.items():
            if catalog.setdefault(key, value) != value:
                raise RuntimeError(f"Conflicting English translations for {key!r}")
    return catalog


_ENGLISH = _load_english()


def english_catalog() -> dict[str, str]:
    """Copy of the Russian-key to English-text catalog (used by the coverage test)."""
    return dict(_ENGLISH)


def current_language() -> str:
    return _LANGUAGE


def set_language(language: str) -> None:
    global _LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}")
    _LANGUAGE = language


def saved_language() -> str:
    """The persisted choice, or English when nothing valid is stored."""
    from PySide6.QtCore import QSettings
    value = str(QSettings().value(_SETTINGS_KEY, DEFAULT_LANGUAGE))
    return value if value in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def initialize_language() -> None:
    """Select the saved language; call after QApplication has its organization/application names."""
    set_language(saved_language())


def save_language(language: str) -> None:
    """Persist a choice for the next start; the running session keeps its language."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}")
    from PySide6.QtCore import QSettings
    QSettings().setValue(_SETTINGS_KEY, language)


def tr(text: str, /, **values: object) -> str:
    """Translate one Russian source string, formatting named placeholders after translation."""
    translated = text if _LANGUAGE == "ru" else _ENGLISH.get(text, text)
    return translated.format(**values) if values else translated
