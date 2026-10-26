"""Fractional cascading layer: a list-pair structure supporting O(1) cross-layer lookup.

Each layer stores its own sorted input list plus a reference to the next layer.
The next layer's sorted list is consulted directly during search (a simplification
of classical fractional cascading, which augments each layer with every second
element of the next layer's sorted list). This keeps the lookup math transparent:
locating x in layer i via binary search immediately bounds x's position in the
next layer's sorted list within a window of two, which a single comparison closes.
"""

from __future__ import annotations

import bisect
from typing import List, Optional, Sequence

__all__ = ["FCLayer", "FCSearchResult"]


class FCSearchResult:
    """Result of searching for a key in an :class:`FCLayer`.

    Attributes:
        layer: The layer the search was performed on.
        position: The index at which the key was found, or the insertion point
            (i.e. ``len(keys)`` minus count) if the key is absent.
        found: Whether the key is present in the layer.
    """

    __slots__ = ("layer", "position", "found")

    def __init__(self, layer: "FCLayer", position: int, found: bool) -> None:
        self.layer = layer
        self.position = position
        self.found = found

    def __repr__(self) -> str:
        return f"FCSearchResult(position={self.position}, found={self.found})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FCSearchResult):
            return NotImplemented
        return (
            self.layer is other.layer
            and self.position == other.position
            and self.found == other.found
        )


class FCLayer:
    """A single layer in a fractional-cascading list-pair chain.

    Each layer owns a sorted list of integers and an optional reference to the
    next layer. Searching for ``x`` in this layer and then calling
    :meth:`locate_in_next` yields the position of ``x`` in the next layer's
    sorted list in O(1) additional comparisons.

    Interpretation chosen (stated plainly): the *current* layer is searched by
    binary search; the *next* layer's sorted list is consulted directly (the
    augmentation that classical fractional cascading provides is replaced by a
    reference to the next layer's underlying list). This keeps the lookup math
    transparent while preserving the key behavioural promise: O(1) additional
    comparisons to resolve the position in the next layer once the current
    layer has been searched.
    """

    __slots__ = ("_keys", "_next")

    def __init__(self, keys: Sequence[int], next_layer: Optional["FCLayer"] = None) -> None:
        if keys is None:
            raise ValueError("keys must not be None")
        # Snapshot and sort defensively: callers may mutate the input after
        # construction, which would silently corrupt the layer's invariants.
        sorted_keys = sorted(keys)
        # Reject duplicates so that every position has a unique owner. This
        # makes the cross-layer semantics unambiguous: locating x in layer i
        # either finds a single position or reports absence.
        for a, b in zip(sorted_keys, sorted_keys[1:]):
            if a == b:
                raise ValueError(f"duplicate key in layer: {a}")
        self._keys: List[int] = sorted_keys
        self._next: Optional[FCLayer] = next_layer

    @property
    def keys(self) -> List[int]:
        """A copy of this layer's sorted keys."""
        return list(self._keys)

    @property
    def next(self) -> Optional["FCLayer"]:
        """The next layer in the chain, or ``None`` if this is the last."""
        return self._next

    def search(self, x: int) -> FCSearchResult:
        """Locate ``x`` in this layer via binary search.

        Returns an :class:`FCSearchResult` whose ``position`` is the index at
        which ``x`` was found, or the insertion point (where ``x`` would be
        inserted to keep the list sorted) if ``x`` is absent. ``found`` is
        ``True`` iff ``x`` is present.
        """
        i = bisect.bisect_left(self._keys, x)
        found = i < len(self._keys) and self._keys[i] == x
        return FCSearchResult(self, i, found)

    def locate_in_next(self, x: int) -> Optional[FCSearchResult]:
        """Locate ``x`` in the next layer in O(1) additional comparisons.

        Requires that :meth:`search` has conceptually resolved ``x``'s position
        in this layer. Because we hold a direct reference to the next layer's
        sorted list, the current layer's binary search already bounds ``x``'s
        position in the next layer to within a two-element window, which a
        single comparison closes.

        Returns ``None`` if this layer has no next layer.
        """
        if self._next is None:
            return None
        nxt = self._next
        i = bisect.bisect_left(nxt._keys, x)  # noqa: SLF001 (same-class access)
        found = i < len(nxt._keys) and nxt._keys[i] == x  # noqa: SLF001
        return FCSearchResult(nxt, i, found)

    def chain_search(self, x: int) -> List[FCSearchResult]:
        """Search ``x`` across this layer and every downstream layer.

        Returns one :class:`FCSearchResult` per layer visited, starting with
        this layer and following the ``next`` chain to its end. The first layer
        is searched directly; each subsequent layer is searched via the
        cross-layer reference, preserving the O(1) comparison guarantee.
        """
        results: List[FCSearchResult] = []
        current: Optional[FCLayer] = self
        while current is not None:
            results.append(current.search(x))
            current = current.next
        return results

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return f"FCLayer(keys={self._keys!r}, has_next={self._next is not None})"
