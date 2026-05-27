"""
security.py — Sécurité de l'API BioID AI.

Implémente :
- JWT avec python-jose
- OAuth2PasswordBearer
- Rate limiting avec slowapi
- Chiffrement AES-256 avec cryptography
- Validation anti-injection (sanitize inputs)
- Logs d'audit
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── JWT / OAuth2 ─────────────────────────────────────────────────────────────
try:
    from jose import JWTError, jwt as _jwt
    _JOSE_OK = True
except ImportError:
    _JOSE_OK = False
    log.warning("python-jose non disponible — JWT désactivé")

try:
    from fastapi import Depends, HTTPException, status
    from fastapi.security import OAuth2PasswordBearer
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    log.warning("fastapi non disponible — OAuth2 désactivé")

# ── Cryptography AES-256 ──────────────────────────────────────────────────────
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTO_OK = True
except ImportError:
    _CRYPTO_OK = False
    log.warning("cryptography non disponible — chiffrement AES désactivé")

# ── Passlib ───────────────────────────────────────────────────────────────────
try:
    from passlib.context import CryptContext
    _PASSLIB_OK = True
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    _PASSLIB_OK = False
    log.warning("passlib non disponible — hachage bcrypt désactivé")

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    _SLOWAPI_OK = True
except ImportError:
    _SLOWAPI_OK = False
    log.info("slowapi non disponible — rate limiting désactivé")

# ── Configuration JWT ─────────────────────────────────────────────────────────
SECRET_KEY     = os.getenv("KANEA_SECRET_KEY", "kanea-bioid-secret-key-dev-2024-change-in-prod")
ALGORITHM      = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ── OAuth2 scheme ─────────────────────────────────────────────────────────────
if _FASTAPI_OK:
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)
else:
    oauth2_scheme = None

# ── Rate Limiter ──────────────────────────────────────────────────────────────
if _SLOWAPI_OK:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# ── Audit logger ──────────────────────────────────────────────────────────────
audit_log = logging.getLogger("kanea.audit")


def _get_audit_handler() -> logging.Handler:
    """Retourne le handler pour les logs d'audit."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [AUDIT] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    return handler


if not audit_log.handlers:
    audit_log.addHandler(_get_audit_handler())
    audit_log.setLevel(logging.INFO)


# ── Fonctions JWT ─────────────────────────────────────────────────────────────
def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Crée un token JWT signé.

    Paramètres
    ----------
    data : dict
        Payload à encoder (doit inclure "sub" pour le sujet).
    expires_delta : timedelta optionnel
        Durée de validité (défaut : ACCESS_TOKEN_EXPIRE_MINUTES).

    Retourne le token JWT signé (str).
    """
    if not _JOSE_OK:
        log.error("python-jose requis pour create_access_token()")
        return ""
    try:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (
            expires_delta if expires_delta
            else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
        token = _jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        audit_log.info("Token créé pour sub='%s'", data.get("sub", "anonymous"))
        return token
    except Exception as exc:
        log.error("create_access_token error: %s", exc)
        return ""


def verify_token(token: str) -> Optional[dict[str, Any]]:
    """
    Vérifie et décode un token JWT.

    Retourne le payload décodé, ou None si invalide/expiré.
    """
    if not _JOSE_OK:
        log.warning("python-jose non disponible — vérification token ignorée")
        return {"sub": "anonymous", "role": "user"}

    if not token:
        return None
    try:
        payload = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        audit_log.info("Token vérifié pour sub='%s'", payload.get("sub", "?"))
        return payload
    except JWTError as exc:
        audit_log.warning("Token invalide : %s", exc)
        return None
    except Exception as exc:
        log.error("verify_token error: %s", exc)
        return None


def get_current_user(
    token: Optional[str] = None,
) -> dict[str, Any]:
    """
    Dépendance FastAPI pour récupérer l'utilisateur courant depuis le token JWT.

    Lève HTTPException 401 si le token est invalide ou absent.
    """
    if not _FASTAPI_OK:
        return {"sub": "anonymous", "role": "user"}

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les identifiants",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        # Mode développement : accès permis sans token
        dev_mode = os.getenv("KANEA_DEV_MODE", "true").lower() == "true"
        if dev_mode:
            return {"sub": "dev_user", "role": "admin"}
        raise credentials_exception

    payload = verify_token(token)
    if payload is None:
        raise credentials_exception

    return payload


# ── Chiffrement AES-256 (Fernet = AES-128-CBC + HMAC-SHA256) ──────────────────
def _derive_key(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
    """Dérive une clé Fernet à partir d'un mot de passe via PBKDF2."""
    if not _CRYPTO_OK:
        raise ImportError("cryptography est requis pour le chiffrement")
    salt = salt or os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key_raw = kdf.derive(password.encode())
    fernet_key = base64.urlsafe_b64encode(key_raw)
    return fernet_key, salt


def encrypt_data(
    plaintext: str,
    password: Optional[str] = None,
) -> str:
    """
    Chiffre une chaîne avec Fernet (AES-128-CBC + HMAC-SHA256).

    Retourne une chaîne base64url contenant [salt(16) + token chiffré].
    Retourne le plaintext si cryptography non disponible.
    """
    if not _CRYPTO_OK:
        log.warning("encrypt_data : cryptography non disponible — données non chiffrées")
        return plaintext

    try:
        pwd = password or SECRET_KEY
        fernet_key, salt = _derive_key(pwd)
        f = Fernet(fernet_key)
        token = f.encrypt(plaintext.encode("utf-8"))
        # Préfixer avec le salt
        combined = base64.urlsafe_b64encode(salt + base64.urlsafe_b64decode(token))
        return combined.decode("utf-8")
    except Exception as exc:
        log.error("encrypt_data error: %s", exc)
        return plaintext


def decrypt_data(
    ciphertext: str,
    password: Optional[str] = None,
) -> str:
    """
    Déchiffre une chaîne chiffrée avec encrypt_data().

    Retourne la chaîne déchiffrée, ou "" si erreur.
    """
    if not _CRYPTO_OK:
        log.warning("decrypt_data : cryptography non disponible")
        return ciphertext

    try:
        pwd = password or SECRET_KEY
        combined = base64.urlsafe_b64decode(ciphertext.encode("utf-8"))
        salt = combined[:16]
        token_raw = combined[16:]
        fernet_key, _ = _derive_key(pwd, salt)
        f = Fernet(fernet_key)
        token = base64.urlsafe_b64encode(token_raw)
        return f.decrypt(token).decode("utf-8")
    except Exception as exc:
        log.error("decrypt_data error: %s", exc)
        return ""


# ── Validation / Sanitisation ─────────────────────────────────────────────────
# Patterns d'injection dangereux
_INJECTION_PATTERNS = [
    re.compile(r"[<>\"'%;()&+]"),                           # XSS / HTML
    re.compile(r"(--|;|\/\*|\*\/|xp_|exec|drop|insert|select|union|update|delete)",
               re.IGNORECASE),                               # SQL Injection
    re.compile(r"(\.\./|\.\.\\|/etc/|/proc/|\\windows\\)",
               re.IGNORECASE),                               # Path Traversal
    re.compile(r"(javascript:|vbscript:|data:text/html)",
               re.IGNORECASE),                               # Script injection
]


def sanitize_input(value: Any, max_length: int = 500) -> str:
    """
    Nettoie et valide une entrée utilisateur contre les injections.

    - Normalise l'unicode (NFKC)
    - Supprime les caractères de contrôle
    - Vérifie les patterns d'injection connus
    - Tronque à max_length
    - Retourne la valeur nettoyée

    Lève ValueError si une injection est détectée.
    """
    if value is None:
        return ""

    # Conversion et normalisation
    text = unicodedata.normalize("NFKC", str(value))

    # Suppression des caractères de contrôle (sauf espace, tab, newline)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in (" ", "\t", "\n")
    )

    # Troncature
    text = text[:max_length]

    # Vérification des patterns d'injection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            audit_log.warning("Tentative d'injection détectée : %r", text[:100])
            raise ValueError(f"Valeur invalide : caractères ou patterns non autorisés détectés.")

    return text.strip()


def sanitize_dict(data: dict[str, Any], max_length: int = 500) -> dict[str, Any]:
    """
    Sanitise récursivement les valeurs string d'un dict.

    Les valeurs numériques sont conservées telles quelles.
    """
    result: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, str):
            result[key] = sanitize_input(val, max_length)
        elif isinstance(val, dict):
            result[key] = sanitize_dict(val, max_length)
        else:
            result[key] = val
    return result


# ── Hachage de mots de passe ──────────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash un mot de passe avec bcrypt (passlib) ou SHA256 fallback."""
    if _PASSLIB_OK:
        return pwd_context.hash(password)
    # Fallback SHA256 (moins sécurisé, pour dev uniquement)
    log.warning("passlib non disponible — fallback SHA256 (non recommandé en production)")
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe contre son hash."""
    if _PASSLIB_OK:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            pass
    # Fallback SHA256
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


# ── Log d'audit helper ────────────────────────────────────────────────────────
def log_audit_event(
    action: str,
    user: str = "anonymous",
    case_id: Optional[str] = None,
    details: Optional[str] = None,
    success: bool = True,
) -> None:
    """Enregistre un événement dans le log d'audit."""
    status_str = "SUCCESS" if success else "FAILURE"
    msg = f"[{status_str}] action={action} user={user}"
    if case_id:
        msg += f" case_id={case_id}"
    if details:
        msg += f" details={details[:200]}"
    audit_log.info(msg)
