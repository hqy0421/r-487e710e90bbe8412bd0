from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Optional, Tuple

from datasketch.hashfunc import sha1_hash64
from datasketch.hyperloglog import HyperLogLogPlusPlus

# A user value of None marks a missing user: the record is tracked but
# contributes nothing to its group's sketch. Any bytes value - including
# b"" - is a valid user.
User = Optional[bytes]


class _Group:
    """Per-group mutable state: exact user reference counts and the sketch."""

    __slots__ = ("sketch", "users")

    def __init__(self, sketch: HyperLogLogPlusPlus) -> None:
        self.users: Dict[bytes, int] = {}  # user -> number of live records referencing it
        self.sketch = sketch


class CorrectableHyperLogLogGroups:
    """Grouped HyperLogLog++ cardinality sketches that support record-level
    corrections and retractions.

    Records are keyed by `record_id` and each record references one user in
    one group (e.g. a channel). Plain HyperLogLog registers can only be
    raised, so a bare ``update`` cannot undo a record that was later
    corrected or retracted. This class keeps, per group, the exact multiset
    of currently referenced users, and maintains the following invariant
    after every mutation:

        For every group, the sketch registers are identical to replaying
        the group's currently referenced users into a fresh
        ``HyperLogLogPlusPlus`` - i.e. identical to a full replay of all
        current records. Estimation therefore uses the unmodified
        HyperLogLog++ ``count``.

    Semantics:

    * ``upsert`` inserts a record or replaces the current version of the
      same ``record_id`` (only the current version is kept), possibly
      changing its user and/or group.
    * ``retract`` removes the current version of a record.
    * A user referenced by several live records is reference-counted:
      removing one record never loses the user's contribution.
    * A record with user ``None`` (missing) is tracked but not counted.
      ``b""`` is a valid, countable user.
    * Mutations only touch the affected group(s): adding a user is one
      HyperLogLog update, and when the last reference to a user leaves a
      group, only that group's sketch is rebuilt by replaying its own
      current users. Other groups and the record table are never scanned.

    Args:
        p (int): The precision parameter of the underlying
            ``HyperLogLogPlusPlus`` sketches.
        hashfunc (Callable): The hash function used by the underlying
            sketches. Defaults to a SHA1-based 64-bit hash, same as
            ``HyperLogLogPlusPlus``.

    Example:
        .. code-block:: python

            groups = CorrectableHyperLogLogGroups(p=12)
            groups.upsert("rec-1", user=b"user-a", group="web")
            groups.upsert("rec-2", user=b"user-b", group="web")
            groups.upsert("rec-1", user=b"user-c", group="app")  # correction
            groups.retract("rec-2")
            groups.count("web")
            sketch = groups.sketch("app")  # independent HyperLogLogPlusPlus

    """

    __slots__ = ("_groups", "_records", "hashfunc", "p")

    def __init__(self, p: int = 8, hashfunc: Callable = sha1_hash64) -> None:
        # Eagerly build one sketch to validate p and hashfunc.
        HyperLogLogPlusPlus(p=p, hashfunc=hashfunc)
        self.p = p
        self.hashfunc = hashfunc
        self._records: Dict[Hashable, Tuple[User, Hashable]] = {}  # record_id -> (user, group)
        self._groups: Dict[Hashable, _Group] = {}  # group -> _Group

    def upsert(self, record_id: Hashable, user: User, group: Hashable) -> bool:
        """Insert a record, or replace the current version of `record_id`.

        Args:
            record_id (Hashable): The record identifier. Only the current
                version of each record is kept.
            user (Optional[bytes]): The user referenced by this record, as
                bytes (bytes-like objects are converted). ``None`` means the
                user is missing: the record is tracked but not counted.
                ``b""`` is a valid user.
            group (Hashable): The group (e.g. channel) this record belongs to.

        Returns:
            bool: True if a previous version of `record_id` was replaced.

        """
        if user is not None and not isinstance(user, bytes):
            user = bytes(user)
        prev = self._records.get(record_id)
        if prev is not None and prev == (user, group):
            return True  # Already the current version: nothing to do.
        if prev is not None and prev[0] is not None:
            self._remove_user(prev[1], prev[0])
        if user is not None:
            self._add_user(group, user)
        self._records[record_id] = (user, group)
        return prev is not None

    def retract(self, record_id: Hashable) -> bool:
        """Remove the current version of a record, if present.

        Args:
            record_id (Hashable): The record identifier to retract.

        Returns:
            bool: True if the record existed and was retracted.

        """
        prev = self._records.pop(record_id, None)
        if prev is None:
            return False
        if prev[0] is not None:
            self._remove_user(prev[1], prev[0])
        return True

    def count(self, group: Hashable) -> float:
        """Estimate the current number of distinct users in a group.

        Args:
            group (Hashable): The group to estimate.

        Returns:
            float: The HyperLogLog++ estimate; 0.0 for an unknown group.

        """
        g = self._groups.get(group)
        return g.sketch.count() if g is not None else 0.0

    def sketch(self, group: Hashable) -> HyperLogLogPlusPlus:
        """Export the group's sketch as an independent HyperLogLogPlusPlus.

        The returned sketch is a native ``HyperLogLogPlusPlus`` that shares
        no state with this container: it can be counted, merged, and
        serialized (``serialize``/``deserialize``, pickling) without
        affecting the stored sketches.

        Args:
            group (Hashable): The group to export.

        Returns:
            HyperLogLogPlusPlus: An independent copy of the group's sketch,
            or an empty sketch if the group is unknown.

        """
        g = self._groups.get(group)
        if g is None:
            return HyperLogLogPlusPlus(p=self.p, hashfunc=self.hashfunc)
        return g.sketch.copy()

    def groups(self) -> List[Hashable]:
        """Returns:
        list: The groups that currently have at least one counted user.

        """
        return list(self._groups)

    def __contains__(self, record_id: Hashable) -> bool:
        return record_id in self._records

    def __len__(self) -> int:
        """Returns:
        int: The number of live records currently tracked.

        """
        return len(self._records)

    def _new_sketch(self) -> HyperLogLogPlusPlus:
        return HyperLogLogPlusPlus(p=self.p, hashfunc=self.hashfunc)

    def _add_user(self, group: Hashable, user: bytes) -> None:
        g = self._groups.get(group)
        if g is None:
            g = _Group(self._new_sketch())
            self._groups[group] = g
        if user in g.users:
            g.users[user] += 1
        else:
            g.users[user] = 1
            g.sketch.update(user)

    def _remove_user(self, group: Hashable, user: bytes) -> None:
        g = self._groups[group]
        remaining = g.users[user] - 1
        if remaining > 0:
            # Other live records still reference this user: its contribution
            # must be kept, and the registers already match a full replay.
            g.users[user] = remaining
            return
        del g.users[user]
        if not g.users:
            # Replay of an empty group is an empty sketch: drop the group.
            del self._groups[group]
            return
        # Registers cannot be lowered, so rebuild this group's sketch by
        # replaying its current users. Only this group is touched.
        sketch = self._new_sketch()
        for u in g.users:
            sketch.update(u)
        g.sketch = sketch
