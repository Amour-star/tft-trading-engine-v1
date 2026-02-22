"""
Security utilities: API key encryption, secret management.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except BaseException:
    Fernet = None  # type: ignore
    HAS_CRYPTOGRAPHY = False


def get_fernet_key() -> bytes:
    """Get or generate a Fernet encryption key."""
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("cryptography package is not available")
    key = os.getenv("FERNET_KEY")
    if key:
        return key.encode()
    # Generate and warn
    new_key = Fernet.generate_key()
    print(f"WARNING: No FERNET_KEY set. Generated: {new_key.decode()}")
    print("Set this in your .env file for persistence.")
    return new_key


_fernet: Optional["Fernet"] = None


def _get_fernet() -> "Fernet":
    global _fernet
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("cryptography package is not available")
    if _fernet is None:
        _fernet = Fernet(get_fernet_key())
    return _fernet


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret string."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a secret string."""
    return _get_fernet().decrypt(ciphertext.encode()).decode()


def mask_key(key: str, visible: int = 4) -> str:
    """Mask an API key for safe display."""
    if len(key) <= visible:
        return "****"
    return key[:visible] + "*" * (len(key) - visible)
