"""
Shared API dependencies: authentication and session management.
"""
import hmac
import threading
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Header, HTTPException, status

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import Settings
from src.rag.memory import ConversationMemory, SessionManager


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Standard auth: open access when no key is configured (local dev),
    constant-time comparison when one is.
    """
    if not Settings.API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, Settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key header",
        )


def verify_admin_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Admin auth: unlike standard endpoints, admin operations (ingest, clear)
    are hard-disabled until a key is configured, so an accidentally exposed
    dev instance cannot be wiped or re-ingested by strangers.
    """
    if not Settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin endpoints disabled: set PLATONICAM_API_KEY to enable",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, Settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key header",
        )


class ApiSessionStore:
    """
    SessionManager wrapper with TTL eviction and a hard session cap.

    Sessions are in-process; running multiple API workers requires sticky
    routing or an external memory store (documented limitation for v1).
    """

    def __init__(
        self,
        ttl_minutes: int = Settings.SESSION_TTL_MINUTES,
        max_sessions: int = Settings.MAX_SESSIONS,
    ):
        self.manager = SessionManager()
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_sessions = max_sessions
        self.last_seen: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> ConversationMemory:
        """Fetch (or create) a session, evicting expired ones first."""
        with self._lock:
            now = datetime.now()

            expired = [sid for sid, ts in self.last_seen.items() if now - ts > self.ttl]
            for sid in expired:
                self.manager.delete_session(sid)
                del self.last_seen[sid]

            if session_id not in self.last_seen and len(self.last_seen) >= self.max_sessions:
                oldest = min(self.last_seen, key=self.last_seen.get)
                self.manager.delete_session(oldest)
                del self.last_seen[oldest]

            self.last_seen[session_id] = now
            return self.manager.get_session(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session; returns True if it existed."""
        with self._lock:
            existed = session_id in self.last_seen
            self.manager.delete_session(session_id)
            self.last_seen.pop(session_id, None)
            return existed

    def active_count(self) -> int:
        with self._lock:
            return len(self.last_seen)
