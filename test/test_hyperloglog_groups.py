import pickle
import random
import unittest
from unittest.mock import patch

from datasketch import CorrectableHyperLogLogGroups, HyperLogLogPlusPlus


def replay_model(model, p):
    """Build the expected sketches by replaying all current records.

    Args:
        model (dict): record_id -> (user, group), where user may be None.
        p (int): The precision parameter.

    Returns:
        dict: group -> HyperLogLogPlusPlus replayed from the model.

    """
    by_group = {}
    for user, group in model.values():
        if user is None:
            continue
        by_group.setdefault(group, []).append(user)
    expected = {}
    for group, users in by_group.items():
        h = HyperLogLogPlusPlus(p=p)
        for u in users:
            h.update(u)
        expected[group] = h
    return expected


class TestCorrectableHyperLogLogGroups(unittest.TestCase):
    def assertConsistent(self, cg, model, p=8):
        """Every group's registers must equal a full replay of current records."""
        expected = replay_model(model, p)
        self.assertEqual(set(cg.groups()), set(expected))
        self.assertEqual(len(cg), len(model))
        for group, h in expected.items():
            self.assertEqual(cg.sketch(group), h)
            self.assertEqual(cg.count(group), h.count())

    def test_upsert_and_count(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        for i in range(100):
            user = b"user-%d" % i
            group = "even" if i % 2 == 0 else "odd"
            cg.upsert("rec-%d" % i, user, group)
            model["rec-%d" % i] = (user, group)
        self.assertConsistent(cg, model)
        self.assertAlmostEqual(cg.count("even"), 50.0, delta=5.0)
        self.assertAlmostEqual(cg.count("odd"), 50.0, delta=5.0)

    def test_correction_same_group(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        for i in range(10):
            cg.upsert(i, b"u%d" % i, "g")
            model[i] = (b"u%d" % i, "g")
        # Correct the user of record 3; only the current version counts.
        cg.upsert(3, b"u-new", "g")
        model[3] = (b"u-new", "g")
        self.assertConsistent(cg, model)
        # Correcting again back to a user that still exists elsewhere.
        cg.upsert(3, b"u0", "g")
        model[3] = (b"u0", "g")
        self.assertConsistent(cg, model)

    def test_correction_moves_group(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        for i in range(10):
            group = "A" if i % 2 == 0 else "B"
            cg.upsert(i, b"u%d" % i, group)
            model[i] = (b"u%d" % i, group)
        # Move record 0 from A to B.
        cg.upsert(0, b"u0", "B")
        model[0] = (b"u0", "B")
        self.assertConsistent(cg, model)
        # Move record 1 from B to A with a new user at the same time.
        cg.upsert(1, b"u-new", "A")
        model[1] = (b"u-new", "A")
        self.assertConsistent(cg, model)

    def test_retract_shared_user_keeps_contribution(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        # Two records referencing the same user in the same group.
        cg.upsert("r1", b"shared", "g")
        model["r1"] = (b"shared", "g")
        cg.upsert("r2", b"shared", "g")
        model["r2"] = (b"shared", "g")
        cg.upsert("r3", b"other", "g")
        model["r3"] = (b"other", "g")
        # Retracting one record must not lose the shared user's contribution.
        before = cg.sketch("g")
        cg.retract("r1")
        del model["r1"]
        self.assertEqual(cg.sketch("g"), before)
        self.assertConsistent(cg, model)
        # Retracting the last reference removes the user.
        cg.retract("r2")
        del model["r2"]
        self.assertConsistent(cg, model)
        # Retracting the last user drops the group; replay is an empty sketch.
        cg.retract("r3")
        del model["r3"]
        self.assertConsistent(cg, model)
        self.assertEqual(cg.count("g"), 0.0)
        self.assertTrue(cg.sketch("g").is_empty())

    def test_missing_user_not_counted(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        # A record with a missing user is tracked but not counted.
        cg.upsert("r1", None, "g")
        model["r1"] = (None, "g")
        self.assertConsistent(cg, model)
        self.assertEqual(cg.count("g"), 0.0)
        self.assertEqual(cg.groups(), [])
        self.assertIn("r1", cg)
        # Correction to a real user makes it counted.
        cg.upsert("r1", b"u1", "g")
        model["r1"] = (b"u1", "g")
        self.assertConsistent(cg, model)
        # Correction back to missing removes the contribution.
        cg.upsert("r1", None, "g")
        model["r1"] = (None, "g")
        self.assertConsistent(cg, model)
        # Retracting a missing-user record changes no sketch.
        cg.retract("r1")
        del model["r1"]
        self.assertConsistent(cg, model)
        self.assertNotIn("r1", cg)

    def test_empty_bytes_user_is_valid(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        cg.upsert("r1", b"", "g")
        model["r1"] = (b"", "g")
        cg.upsert("r2", None, "g")  # missing, not counted
        model["r2"] = (None, "g")
        self.assertConsistent(cg, model)
        self.assertEqual(cg.groups(), ["g"])
        self.assertAlmostEqual(cg.count("g"), 1.0, delta=0.5)
        # b"" is distinct from missing: retracting the missing-user record
        # keeps it, retracting its own record removes it.
        cg.retract("r2")
        del model["r2"]
        self.assertConsistent(cg, model)
        self.assertEqual(cg.groups(), ["g"])
        cg.retract("r1")
        del model["r1"]
        self.assertConsistent(cg, model)
        self.assertEqual(cg.groups(), [])

    def test_bytearray_user_normalized(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        cg.upsert("r1", bytearray(b"u1"), "g")
        cg.upsert("r2", memoryview(b"u1"), "g")
        model = {"r1": (b"u1", "g"), "r2": (b"u1", "g")}
        self.assertConsistent(cg, model)
        # Same value as bytes is a no-op correction, not a second user.
        cg.upsert("r1", b"u1", "g")
        self.assertConsistent(cg, model)
        cg.retract("r1")
        del model["r1"]
        self.assertConsistent(cg, model)

    def test_return_values(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        self.assertFalse(cg.upsert("r1", b"u1", "g"))  # new record
        self.assertTrue(cg.upsert("r1", b"u1", "g"))  # no-op correction
        self.assertTrue(cg.upsert("r1", b"u2", "g"))  # correction
        self.assertFalse(cg.retract("unknown"))
        self.assertTrue(cg.retract("r1"))
        self.assertFalse(cg.retract("r1"))

    def test_unknown_group(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        self.assertEqual(cg.count("nope"), 0.0)
        sketch = cg.sketch("nope")
        self.assertIsInstance(sketch, HyperLogLogPlusPlus)
        self.assertTrue(sketch.is_empty())

    def test_randomized_matches_full_replay(self):
        p = 8
        cg = CorrectableHyperLogLogGroups(p=p)
        rng = random.Random(20261116)
        users = [b"u%d" % i for i in range(40)] + [b""]
        groups = ["g%d" % i for i in range(5)]
        rids = ["r%d" % i for i in range(120)]
        model = {}
        for _ in range(3000):
            if model and rng.random() < 0.35:
                rid = rng.choice(rids)
                cg.retract(rid)
                model.pop(rid, None)
            else:
                rid = rng.choice(rids)
                user = None if rng.random() < 0.15 else rng.choice(users)
                group = rng.choice(groups)
                cg.upsert(rid, user, group)
                model[rid] = (user, group)
            self.assertConsistent(cg, model, p)

    def test_export_is_independent(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        cg.upsert("r1", b"u1", "g")
        cg.upsert("r2", b"u2", "g")
        before = cg.sketch("g")
        exported = cg.sketch("g")
        self.assertIsInstance(exported, HyperLogLogPlusPlus)
        self.assertEqual(exported, before)
        # Mutating the exported sketch must not affect the stored one.
        exported.update(b"intruder")
        other = HyperLogLogPlusPlus(p=8)
        other.update(b"another-intruder")
        exported.merge(other)
        self.assertEqual(cg.sketch("g"), before)

    def test_export_serialization(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        cg.upsert("r1", b"u1", "g")
        cg.upsert("r2", b"u2", "g")
        sketch = cg.sketch("g")
        # serialize/deserialize roundtrip of the native sketch.
        buf = bytearray(sketch.bytesize())
        sketch.serialize(buf)
        self.assertEqual(HyperLogLogPlusPlus.deserialize(buf), sketch)
        # Pickle roundtrip.
        self.assertEqual(pickle.loads(pickle.dumps(sketch)), sketch)

    def test_export_merge(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        model = {}
        for i, (user, group) in enumerate([(b"u1", "A"), (b"u2", "A"), (b"u3", "A"), (b"u3", "B"), (b"u4", "B")]):
            cg.upsert(i, user, group)
            model[i] = (user, group)
        # Merging exported sketches gives the union, as a native sketch.
        merged = cg.sketch("A")
        merged.merge(cg.sketch("B"))
        expected = HyperLogLogPlusPlus(p=8)
        for user in (b"u1", b"u2", b"u3", b"u4"):
            expected.update(user)
        self.assertEqual(merged, expected)
        self.assertEqual(HyperLogLogPlusPlus.union(cg.sketch("A"), cg.sketch("B")), expected)
        # The stored sketches are unchanged by the exported merge.
        self.assertConsistent(cg, model)

    def test_mutation_touches_only_affected_group(self):
        cg = CorrectableHyperLogLogGroups(p=8)
        cg.upsert("r1", b"u1", "A")
        cg.upsert("r2", b"u2", "A")
        cg.upsert("r3", b"u2", "A")
        cg.upsert("r4", b"u3", "B")
        cg.upsert("r5", b"u4", "B")
        sketch_a = cg._groups["A"].sketch
        sketch_b = cg._groups["B"].sketch

        calls = []
        orig_update = HyperLogLogPlusPlus.update

        def spy(self, b):
            calls.append(self)
            return orig_update(self, b)

        # Retracting r1 removes the last reference to u1 in A: only A is
        # rebuilt, replaying its single remaining user.
        with patch.object(HyperLogLogPlusPlus, "update", spy):
            cg.retract("r1")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], cg._groups["A"].sketch)
        self.assertIsNot(cg._groups["A"].sketch, sketch_a)
        self.assertIs(cg._groups["B"].sketch, sketch_b)

        # A no-op correction performs no sketch updates at all.
        del calls[:]
        with patch.object(HyperLogLogPlusPlus, "update", spy):
            cg.upsert("r2", b"u2", "A")
        self.assertEqual(calls, [])

        # Correcting r2's user within A: u2 is still referenced by r3, so no
        # rebuild; exactly one update for the new user, on A's sketch only.
        del calls[:]
        sketch_a = cg._groups["A"].sketch
        with patch.object(HyperLogLogPlusPlus, "update", spy):
            cg.upsert("r2", b"u5", "A")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], sketch_a)
        self.assertIs(cg._groups["A"].sketch, sketch_a)
        self.assertIs(cg._groups["B"].sketch, sketch_b)

        # Moving r4 from B to A touches both groups but nothing else:
        # one update replaying u4 into B's new sketch, one adding u3 to A.
        del calls[:]
        sketch_a = cg._groups["A"].sketch
        with patch.object(HyperLogLogPlusPlus, "update", spy):
            cg.upsert("r4", b"u3", "A")
        self.assertEqual(len(calls), 2)
        self.assertIs(cg._groups["B"].sketch, calls[0])
        self.assertIsNot(cg._groups["B"].sketch, sketch_b)
        self.assertIs(calls[1], sketch_a)
        self.assertIs(cg._groups["A"].sketch, sketch_a)

    def test_accuracy_sanity(self):
        cg = CorrectableHyperLogLogGroups(p=14)
        model = {}
        for i in range(5000):
            user = b"user-%d" % i
            cg.upsert(i, user, "g")
            model[i] = (user, "g")
        self.assertAlmostEqual(cg.count("g"), 5000.0, delta=5000.0 * 0.05)
        # Retract half of the records; the estimate must follow the replay.
        for i in range(2500):
            cg.retract(i)
            del model[i]
        self.assertConsistent(cg, model, p=14)
        self.assertAlmostEqual(cg.count("g"), 2500.0, delta=2500.0 * 0.05)


if __name__ == "__main__":
    unittest.main()
