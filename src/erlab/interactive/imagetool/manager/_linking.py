"""Linker registry for ImageTool manager windows."""

from __future__ import annotations

__all__ = ["_ManagerLinkRegistry"]

import typing

if typing.TYPE_CHECKING:
    import erlab.interactive.imagetool.viewer_linking


class _ManagerLinkRegistry:
    """Own ImageTool linker proxies and deferred cleanup state."""

    def __init__(self) -> None:
        self._linkers: list[
            erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy
        ] = []
        self._cleanup_pending: bool = False

    @property
    def linkers(
        self,
    ) -> tuple[erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy, ...]:
        return tuple(self._linkers)

    def append(
        self, linker: erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy
    ) -> None:
        self._linkers.append(linker)

    def index(
        self, linker: erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy
    ) -> int:
        return self._linkers.index(linker)

    def request_cleanup(self, *, defer: bool) -> bool:
        if defer:
            self._cleanup_pending = True
            return False
        self.cleanup_stale()
        return True

    def pop_pending_cleanup(self) -> bool:
        if not self._cleanup_pending:
            return False
        self._cleanup_pending = False
        return True

    def clear_pending_cleanup(self) -> None:
        self._cleanup_pending = False

    def cleanup_stale(self) -> None:
        for linker in list(self._linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self._linkers.remove(linker)
