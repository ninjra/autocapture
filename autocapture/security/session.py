"""Local security session manager for unlock gating."""

from __future__ import annotations

import datetime as dt
import os
import secrets
import sys
from dataclasses import dataclass
from typing import Protocol

from ..logging_utils import get_logger


class UnlockProvider(Protocol):
    def verify(self) -> bool: ...


@dataclass(frozen=True)
class SecuritySession:
    token: str
    expires_at: dt.datetime


class SecuritySessionManager:
    def __init__(
        self,
        *,
        ttl_seconds: int,
        provider: str,
        test_mode_bypass: bool = False,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._provider_name = provider
        self._tokens: dict[str, dt.datetime] = {}
        self._log = get_logger("security.session")
        self._test_mode_bypass = test_mode_bypass

    @property
    def provider(self) -> str:
        return self._provider_name

    def unlock(self) -> SecuritySession | None:
        if not self._verify_provider():
            return None
        token = secrets.token_urlsafe(48)
        expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=self._ttl_seconds)
        self._tokens[token] = expires_at
        return SecuritySession(token=token, expires_at=expires_at)

    def lock(self) -> int:
        count = len(self._tokens)
        self._tokens.clear()
        return count

    def is_unlocked(self, token: str | None) -> bool:
        if self._test_mode_bypass:
            return True
        if not token:
            return False
        now = dt.datetime.now(dt.timezone.utc)
        expires_at = self._tokens.get(token)
        if not expires_at:
            return False
        if expires_at <= now:
            self._tokens.pop(token, None)
            return False
        return True

    def _verify_provider(self) -> bool:
        provider = self._provider_name
        if provider == "test":
            return True
        if provider == "disabled":
            return False
        if sys.platform != "win32":
            self._log.warning("Security provider %s unavailable on this platform", provider)
            return False
        if provider == "windows_hello":
            if _verify_windows_hello():
                return True
            self._log.info("Windows Hello unavailable; falling back to credential prompt")
            return _verify_cred_ui()
        if provider == "cred_ui":
            return _verify_cred_ui()
        self._log.warning("Unknown security provider: %s", provider)
        return False


def _verify_windows_hello() -> bool:
    try:
        from winrt.windows.security.credentials.ui import (  # type: ignore
            UserConsentVerificationResult,
            UserConsentVerifier,
            UserConsentVerifierAvailability,
        )
    except Exception:
        return False

    try:
        availability = UserConsentVerifier.check_availability_async().get()
        if availability != UserConsentVerifierAvailability.AVAILABLE:
            return False
        prompt = "Unlock Autocapture"
        result = UserConsentVerifier.request_verification_async(prompt).get()
        return result == UserConsentVerificationResult.VERIFIED
    except Exception:
        return False


def _verify_cred_ui() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False

    log = get_logger("security.cred_ui")

    class CREDUI_INFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("hwndParent", wintypes.HWND),
            ("pszMessageText", wintypes.LPCWSTR),
            ("pszCaptionText", wintypes.LPCWSTR),
            ("hbmBanner", wintypes.HANDLE),
        ]

    credui = ctypes.windll.credui
    advapi32 = ctypes.windll.advapi32
    CREDUIWIN_GENERIC = 0x1
    CREDUIWIN_CHECKBOX = 0x2
    LOGON32_LOGON_INTERACTIVE = 2
    LOGON32_PROVIDER_DEFAULT = 0
    CRED_PACK_PROTECTED_CREDENTIALS = 0x1

    info = CREDUI_INFO()
    info.cbSize = ctypes.sizeof(info)
    info.pszCaptionText = "Autocapture Unlock"
    info.pszMessageText = "Verify your Windows credentials to unlock Autocapture."

    auth_buffer = ctypes.c_void_p()
    auth_buffer_size = wintypes.ULONG(0)
    save = wintypes.BOOL(False)

    result = credui.CredUIPromptForWindowsCredentialsW(
        ctypes.byref(info),
        0,
        None,
        None,
        0,
        ctypes.byref(auth_buffer),
        ctypes.byref(auth_buffer_size),
        ctypes.byref(save),
        CREDUIWIN_GENERIC | CREDUIWIN_CHECKBOX,
    )
    if result != 0:
        return False

    try:
        username = ctypes.create_unicode_buffer(256)
        password = ctypes.create_unicode_buffer(256)
        domain = ctypes.create_unicode_buffer(256)
        username_len = wintypes.DWORD(256)
        password_len = wintypes.DWORD(256)
        domain_len = wintypes.DWORD(256)

        unpacked = credui.CredUnPackAuthenticationBufferW(
            CRED_PACK_PROTECTED_CREDENTIALS,
            auth_buffer,
            auth_buffer_size,
            username,
            ctypes.byref(username_len),
            domain,
            ctypes.byref(domain_len),
            password,
            ctypes.byref(password_len),
        )
        if not unpacked:
            return False

        user_value = username.value
        domain_value = domain.value or "."
        if "\\" in user_value:
            domain_value, user_value = user_value.split("\\", 1)
        elif "@" in user_value:
            user_value, domain_value = user_value.split("@", 1)

        token_handle = wintypes.HANDLE()
        logged_in = advapi32.LogonUserW(
            user_value,
            domain_value,
            password.value,
            LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT,
            ctypes.byref(token_handle),
        )
        if logged_in:
            ctypes.windll.kernel32.CloseHandle(token_handle)
        return bool(logged_in)
    except Exception as exc:
        log.warning("Credential verification failed: {}", exc)
        return False
    finally:
        try:
            credui.CredFree(auth_buffer)
        except Exception:
            pass


def is_test_mode() -> bool:
    value = os.environ.get("AUTOCAPTURE_TEST_MODE")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))
