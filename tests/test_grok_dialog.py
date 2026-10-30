"""Regression: delayed Qt destruction must not permit duplicate request dialogs."""
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from barforge.grok_ui import GrokController
from barforge.xai_oauth import OAuthManager, OAuthTokens, OAuthTokenStore


class EmptyVault:
    def get_password(self, *args):
        return None


class GrokDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reopen_before_deferred_delete_keeps_one_active_dialog(self):
        with tempfile.TemporaryDirectory() as folder:
            tokens = OAuthTokenStore(EmptyVault(), "test")
            tokens.save(OAuthTokens("access-test", "refresh-test", time.time() + 3600), persist=False)
            controller = GrokController(settings_path=Path(folder) / "grok.json",
                                        oauth=OAuthManager(tokens))
            controller.apply_settings(enabled=True, model="grok-4.7")
            image = Image.new("RGB", (64, 64), "white")
            callback = lambda _value: True
            first = controller.ask(None, image, "first", callback, callback)
            first.reject()
            second = controller.ask(None, image, "second", callback, callback)
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            repeated = controller.ask(None, image, "third", callback, callback)
            try:
                self.assertIs(repeated, second, "An already open request dialog must be reused")
            finally:
                repeated.reject()
                if repeated is not second:
                    second.reject()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                controller.shutdown()


if __name__ == "__main__":
    unittest.main()
