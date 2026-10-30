"""OAuth security/state regressions; no real credentials or provider traffic."""
import concurrent.futures
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx

from barforge.grok import GrokSettings, SettingsStore
from barforge.xai_oauth import OAuthCancelled, OAuthError, OAuthManager, OAuthTokens, OAuthTokenStore


DISCOVERY = {
    "issuer": "https://auth.x.ai",
    "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
    "token_endpoint": "https://auth.x.ai/oauth2/token",
    "revocation_endpoint": "https://auth.x.ai/oauth2/revoke",
}


class MemoryVault:
    def __init__(self):
        self.value = None

    def get_password(self, *_args):
        return self.value

    def set_password(self, _service, _user, value):
        self.value = value

    def delete_password(self, *_args):
        self.value = None


class Provider:
    def __init__(self, token_handler=None, revoke_status=200):
        self.token_handler = token_handler
        self.revoke_status = revoke_status

    def __call__(self, request):
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json=DISCOVERY)
        if request.url.path == "/oauth2/device/code":
            return httpx.Response(200, json={
                "device_code": "private-device-challenge", "user_code": "ABCD-EFGH",
                "verification_uri": "https://accounts.x.ai/device",
                "verification_uri_complete": "https://accounts.x.ai/device?user_code=ABCD-EFGH",
                "expires_in": 60, "interval": 0.01,
            })
        if request.url.path == "/oauth2/token" and self.token_handler:
            return self.token_handler(request)
        if request.url.path == "/oauth2/revoke":
            return httpx.Response(self.revoke_status, json={"error": "temporarily_unavailable"})
        raise AssertionError(f"Unexpected endpoint: {request.url.path}")


def token_reply(access="new-access", refresh="new-refresh"):
    return httpx.Response(200, json={
        "access_token": access, "refresh_token": refresh, "token_type": "Bearer",
        "expires_in": 3600, "scope": "openid offline_access api:access",
    })


class OAuthBoundaryTests(unittest.TestCase):
    def make_manager(self, provider, *, clock=time.time):
        vault = MemoryVault()
        store = OAuthTokenStore(vault, "test vault")
        manager = OAuthManager(store, transport=httpx.MockTransport(provider), clock=clock)
        return manager, store, vault

    def test_cancel_during_approval_discards_late_tokens(self):
        stopped = threading.Event()

        def approved(_request):
            stopped.set()  # Response arrived after the user closed/cancelled the login.
            return token_reply()

        manager, store, vault = self.make_manager(Provider(approved))
        device = manager.begin_login(stopped)
        with self.assertRaises(OAuthCancelled):
            manager.complete_login(device, stopped, persist=True)
        self.assertFalse(manager.has_session)
        self.assertIsNone(store.load())
        self.assertIsNone(vault.value)

    def test_rotating_refresh_is_single_flight_and_old_401_cannot_clear_it(self):
        refreshes = []
        guard = threading.Lock()

        def rotated(request):
            fields = parse_qs(request.content.decode())
            with guard:
                refreshes.append(fields.get("refresh_token", [""])[0])
                accepted = len(refreshes) == 1 and refreshes[0] == "old-refresh"
            if not accepted:
                return httpx.Response(400, json={"error": "invalid_grant"})
            time.sleep(0.03)  # Keep the response in flight while the competing caller enters.
            return token_reply("rotated-access", "rotated-refresh")

        manager, store, vault = self.make_manager(Provider(rotated))
        store.save(OAuthTokens("old-access", "old-refresh", time.time() - 100), persist=True)
        manager = OAuthManager(store, transport=httpx.MockTransport(Provider(rotated)))
        barrier = threading.Barrier(3)

        def obtain():
            barrier.wait(timeout=3)
            return manager.get_access_token()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            calls = [pool.submit(obtain) for _ in range(2)]
            barrier.wait(timeout=3)
            self.assertEqual([call.result(timeout=5) for call in calls], ["rotated-access"] * 2)
        self.assertEqual(refreshes, ["old-refresh"])
        reloaded = OAuthTokenStore(vault, "test vault").load()
        self.assertEqual(reloaded.refresh_token, "rotated-refresh")
        manager.invalidate(access_token="old-access")
        self.assertTrue(manager.has_session)
        self.assertEqual(manager.get_access_token(), "rotated-access")

    def test_invalid_refresh_forgets_credentials_without_echoing_secrets(self):
        def refused(_request):
            return httpx.Response(400, json={
                "error": "invalid_grant", "error_description": "private-refresh private-access",
            })

        manager, store, vault = self.make_manager(Provider(refused))
        store.save(OAuthTokens("private-access", "private-refresh", time.time() - 100), persist=True)
        manager = OAuthManager(store, transport=httpx.MockTransport(Provider(refused)))
        with self.assertRaises(OAuthError) as failure:
            manager.get_access_token()
        self.assertNotIn("private-refresh", str(failure.exception))
        self.assertNotIn("private-access", str(failure.exception))
        self.assertFalse(manager.has_session)
        self.assertIsNone(vault.value)

    def test_logout_forgets_locally_even_when_remote_revoke_fails(self):
        manager, store, vault = self.make_manager(Provider(revoke_status=503))
        store.save(OAuthTokens("logout-access", "logout-refresh", time.time() + 3600), persist=True)
        manager = OAuthManager(store, transport=httpx.MockTransport(Provider(revoke_status=503)))
        with self.assertRaises(OAuthError):
            manager.logout()
        self.assertFalse(manager.has_session)
        self.assertIsNone(store.load())
        self.assertIsNone(vault.value)
        with self.assertRaises(OAuthError):
            manager.get_access_token()

    def test_discovery_cannot_redirect_credentials_to_another_origin(self):
        destinations = []

        def malicious(request):
            destinations.append(str(request.url))
            if request.url.host == "auth.x.ai":
                return httpx.Response(200, json={**DISCOVERY,
                    "token_endpoint": "https://attacker.invalid/oauth2/token"})
            raise AssertionError("OAuth followed an untrusted discovery endpoint")

        manager, _store, _vault = self.make_manager(malicious)
        with self.assertRaises(OAuthError):
            manager.begin_login()
        self.assertEqual(destinations, ["https://auth.x.ai/.well-known/openid-configuration"])
        self.assertFalse(manager.has_session)

    def test_slow_down_delays_following_poll_and_then_completes(self):
        now = [1000.0]
        poll_times = []

        def pending(_request):
            poll_times.append(now[0])
            if len(poll_times) == 1:
                return httpx.Response(400, json={"error": "authorization_pending"})
            if len(poll_times) == 2:
                return httpx.Response(400, json={"error": "slow_down"})
            return token_reply()

        manager, _store, _vault = self.make_manager(Provider(pending), clock=lambda: now[0])
        stopped = threading.Event()
        device = manager.begin_login(stopped)

        def advance(seconds):
            now[0] += seconds
            return stopped.is_set()

        with patch.object(stopped, "wait", side_effect=advance):
            manager.complete_login(device, stopped, persist=False)
        self.assertGreaterEqual(poll_times[2] - poll_times[1], 5)
        self.assertTrue(manager.has_session)
        self.assertEqual(manager.get_access_token(), "new-access")

    def test_device_expiring_during_wait_cannot_finish_login(self):
        now = [1000.0]
        manager, store, vault = self.make_manager(Provider(lambda _request: token_reply()),
                                                 clock=lambda: now[0])
        stopped = threading.Event()
        device = manager.begin_login(stopped)

        def expire(_seconds):
            now[0] = device.expires_at
            return False

        with patch.object(stopped, "wait", side_effect=expire), self.assertRaises(OAuthError):
            manager.complete_login(device, stopped, persist=True)
        self.assertFalse(manager.has_session)
        self.assertIsNone(store.load())
        self.assertIsNone(vault.value)

    def test_tokens_persist_only_in_vault_not_settings_or_representations(self):
        def renew(request):
            fields = parse_qs(request.content.decode())
            if fields.get("refresh_token") != ["secret-refresh-marker"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return token_reply("restored-access", "restored-refresh")

        _manager, store, vault = self.make_manager(Provider(renew))
        tokens = OAuthTokens("secret-access-marker", "secret-refresh-marker", time.time() + 3600)
        store.save(tokens, persist=True)
        reopened = OAuthTokenStore(vault, "test vault")
        restored = reopened.load()
        self.assertNotIn(tokens.access_token, vault.value)
        self.assertEqual(restored.refresh_token, tokens.refresh_token)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "grok.json"
            SettingsStore(path).save(GrokSettings(enabled=True, model="grok-4.7"))
            visible = path.read_text(encoding="utf-8") + repr(tokens) + repr(restored)
        self.assertNotIn(tokens.access_token, visible)
        self.assertNotIn(tokens.refresh_token, visible)
        restarted = OAuthManager(reopened, transport=httpx.MockTransport(Provider(renew)))
        self.assertEqual(restarted.get_access_token(), "restored-access")
        store.clear()
        self.assertIsNone(OAuthTokenStore(vault, "test vault").load())


if __name__ == "__main__":
    unittest.main()
