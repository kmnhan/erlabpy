"""HMAC and SQLite notary for executable-content manifests."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import os
import pathlib
import secrets
import sqlite3
import tempfile
import time
import typing

if typing.TYPE_CHECKING:
    from erlab.interactive._code_trust._core import CodeTrustManifest

logger = logging.getLogger(__name__)

_ALGORITHM = "sha256"
_SECRET_BYTES = 1024
_STORE_SCHEMA_VERSION = 1
_DEFAULT_CACHE_SIZE = 65535


class CodeTrustError(RuntimeError):
    """Raised when durable code-trust state cannot be accessed safely."""


class CodeTrustNotary:
    """Sign and verify code manifests with one per-user secret."""

    def __init__(
        self,
        storage_directory: str | os.PathLike[str],
        *,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        if cache_size < 1:
            raise ValueError("Code trust cache size must be positive")
        self._storage_directory = pathlib.Path(storage_directory)
        self._secret_path = self._storage_directory / "code_trust_secret"
        self._database_path = self._storage_directory / "code_signatures.db"
        self._cache_size = cache_size

    def _prepare_directory(self) -> None:
        try:
            self._storage_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.name == "posix":
                self._storage_directory.chmod(0o700)
        except OSError as exc:
            raise CodeTrustError(
                f"Could not prepare code trust directory {self._storage_directory}"
            ) from exc

    def _secret(self) -> bytes:
        self._prepare_directory()
        if not self._secret_path.exists():
            self._create_secret()
        try:
            secret = self._secret_path.read_bytes()
            if len(secret) != _SECRET_BYTES:
                raise CodeTrustError("Code trust secret has an invalid size")
            if os.name == "posix":
                self._secret_path.chmod(0o600)
        except OSError as exc:
            raise CodeTrustError(
                f"Could not read code trust secret {self._secret_path}"
            ) from exc
        else:
            return secret

    def _create_secret(self) -> None:
        """Atomically publish one complete secret without replacing a race winner."""
        descriptor = -1
        temporary_path: pathlib.Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".code_trust_secret-",
                dir=self._storage_directory,
            )
            temporary_path = pathlib.Path(temporary_name)
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            stream = os.fdopen(descriptor, "wb")
            descriptor = -1
            with stream:
                stream.write(secrets.token_bytes(_SECRET_BYTES))
                stream.flush()
                os.fsync(stream.fileno())
            # Another process can publish its complete secret first.
            with contextlib.suppress(FileExistsError):
                os.link(temporary_path, self._secret_path)
        except OSError as exc:
            raise CodeTrustError(
                f"Could not create code trust secret {self._secret_path}"
            ) from exc
        finally:
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()

    def _connect(self) -> sqlite3.Connection:
        self._prepare_directory()
        try:
            connection = sqlite3.connect(self._database_path)
        except sqlite3.Error as exc:
            raise CodeTrustError(
                f"Could not access code trust database {self._database_path}"
            ) from exc
        try:
            self._initialize_database(connection)
        except CodeTrustError:
            connection.close()
            raise
        except sqlite3.OperationalError as exc:
            connection.close()
            raise CodeTrustError(
                f"Could not access code trust database {self._database_path}"
            ) from exc
        except sqlite3.DatabaseError:
            connection.close()
            connection = self._replace_unreadable_database()
        try:
            if os.name == "posix":
                self._database_path.chmod(0o600)
        except OSError as exc:
            connection.close()
            raise CodeTrustError(
                f"Could not protect code trust database {self._database_path}"
            ) from exc
        return connection

    def _initialize_database(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=DELETE")
        schema_version = typing.cast(
            "tuple[int]", connection.execute("PRAGMA user_version").fetchone()
        )[0]
        if schema_version not in {0, _STORE_SCHEMA_VERSION}:
            raise CodeTrustError(
                f"Unsupported code trust store schema {schema_version}"
            )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS signatures ("
            "domain TEXT NOT NULL, algorithm TEXT NOT NULL, "
            "signature TEXT NOT NULL, last_seen REAL NOT NULL, "
            "PRIMARY KEY (domain, algorithm, signature))"
        )
        if schema_version == 0:
            connection.execute(f"PRAGMA user_version={_STORE_SCHEMA_VERSION}")
        connection.commit()

    def _replace_unreadable_database(self) -> sqlite3.Connection:
        """Quarantine an unreadable store and create an empty one, as Jupyter does."""
        backup_path = self._database_path.with_name(f"{self._database_path.name}.bak")
        connection: sqlite3.Connection | None = None
        try:
            self._database_path.replace(backup_path)
            connection = sqlite3.connect(self._database_path)
            self._initialize_database(connection)
            if os.name == "posix":
                self._database_path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            if connection is not None:
                connection.close()
            raise CodeTrustError(
                f"Could not recover code trust database {self._database_path}"
            ) from exc
        logger.warning(
            "The code trust database was unreadable. It was moved to %s and "
            "replaced with an empty database.",
            backup_path,
        )
        return connection

    def _signature(self, manifest: CodeTrustManifest) -> str:
        return hmac.new(
            self._secret(), manifest.canonical_bytes(), hashlib.sha256
        ).hexdigest()

    def check(self, manifest: CodeTrustManifest) -> bool:
        """Return whether the current user previously signed the manifest."""
        if not manifest.has_executable_code:
            return True
        try:
            expected = self._signature(manifest)
            with contextlib.closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT signature FROM signatures "
                    "WHERE domain = ? AND algorithm = ? AND signature = ?",
                    (manifest.domain, _ALGORITHM, expected),
                ).fetchone()
                if row is None or not hmac.compare_digest(row[0], expected):
                    return False
                connection.execute(
                    "UPDATE signatures SET last_seen = ? "
                    "WHERE domain = ? AND algorithm = ? AND signature = ?",
                    (time.time(), manifest.domain, _ALGORITHM, expected),
                )
                connection.commit()
        except (CodeTrustError, sqlite3.Error):
            logger.warning("Could not verify saved code trust", exc_info=True)
            return False
        else:
            return True

    def sign(self, manifest: CodeTrustManifest) -> None:
        """Remember trust for one executable manifest."""
        if not manifest.has_executable_code:
            return
        signature = self._signature(manifest)
        try:
            with contextlib.closing(self._connect()) as connection:
                connection.execute(
                    "INSERT INTO signatures "
                    "(domain, algorithm, signature, last_seen) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(domain, algorithm, signature) "
                    "DO UPDATE SET last_seen = excluded.last_seen",
                    (manifest.domain, _ALGORITHM, signature, time.time()),
                )
                self._cull(connection)
                connection.commit()
        except sqlite3.Error as exc:
            raise CodeTrustError("Could not store code trust signature") from exc

    def remove(self, manifest: CodeTrustManifest) -> None:
        """Remove saved trust for one manifest."""
        if not manifest.has_executable_code:
            return
        signature = self._signature(manifest)
        try:
            with contextlib.closing(self._connect()) as connection:
                connection.execute(
                    "DELETE FROM signatures "
                    "WHERE domain = ? AND algorithm = ? AND signature = ?",
                    (manifest.domain, _ALGORITHM, signature),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise CodeTrustError("Could not remove code trust signature") from exc

    def reset(self, *, domain: str | None = None) -> None:
        """Remove all signatures, optionally limited to one trust domain."""
        try:
            with contextlib.closing(self._connect()) as connection:
                if domain is None:
                    connection.execute("DELETE FROM signatures")
                else:
                    connection.execute(
                        "DELETE FROM signatures WHERE domain = ?", (domain,)
                    )
                connection.commit()
        except sqlite3.Error as exc:
            raise CodeTrustError("Could not reset saved code trust") from exc

    def _cull(self, connection: sqlite3.Connection) -> None:
        (count,) = typing.cast(
            "tuple[int]",
            connection.execute("SELECT COUNT(*) FROM signatures").fetchone(),
        )
        if count <= self._cache_size:
            return
        keep = max(int(0.75 * self._cache_size), 1)
        connection.execute(
            "DELETE FROM signatures WHERE rowid IN ("
            "SELECT rowid FROM signatures ORDER BY last_seen DESC LIMIT -1 OFFSET ?)",
            (keep,),
        )
