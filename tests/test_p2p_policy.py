"""Execute the actual policy functions with CPU driver/collective doubles.

No torch/CUDA import or GPU payload is required; these tests cover capability
iteration and group agreement, not kernel correctness or physical bandwidth.
"""
import ast
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def source_function(filename, name, namespace, class_name=None):
    tree = ast.parse((ROOT / filename).read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class P2PPolicy(unittest.TestCase):
    def test_skip_check_visits_every_peer_with_visible_mapping(self):
        seen = []
        mapping = [3, 1, 0, 2]
        def capability(src, dst):
            seen.append((src, dst))
            return dst != 2
        ns = {"envs": SimpleNamespace(VLLM_SKIP_P2P_CHECK=True), "logger": logging.getLogger("test"),
              "torch": SimpleNamespace(cuda=SimpleNamespace(can_device_access_peer=capability)),
              "current_platform": SimpleNamespace(logical_device_id_to_visible_device_id=lambda i: mapping[i])}
        fn = source_function("vllm/distributed/device_communicators/custom_all_reduce.py", "_can_p2p", ns)
        self.assertFalse(fn(0, 4))
        self.assertEqual(seen, [(3, 1), (3, 0), (3, 2)])

    def test_full_payload_check_and_all_supported_case(self):
        for skip in [False, True]:
            calls = []
            def check(a, b):
                calls.append((a, b))
                return True
            ns = {"envs": SimpleNamespace(VLLM_SKIP_P2P_CHECK=skip), "logger": logging.getLogger("test"),
                  "torch": SimpleNamespace(cuda=SimpleNamespace(can_device_access_peer=check)),
                  "gpu_p2p_access_check": check,
                  "current_platform": SimpleNamespace(logical_device_id_to_visible_device_id=lambda i: i)}
            fn = source_function("vllm/distributed/device_communicators/custom_all_reduce.py", "_can_p2p", ns)
            for rank in range(4):
                self.assertTrue(fn(rank, 4))
            self.assertEqual(len(calls), 12)
            self.assertEqual(set(calls), {(a, b) for a in range(4) for b in range(4) if a != b})

    def select(self, local, remote, preference="auto", group_present=True):
        votes = []
        class Tensor:
            def __init__(self, data):
                self.value = data[0]
            def item(self):
                return self.value
        torch = ModuleType("torch")
        torch.__path__ = []
        torch.int32 = "int32"
        torch.tensor = lambda data, **kwargs: Tensor(data)
        dist = ModuleType("torch.distributed")
        dist.ReduceOp = SimpleNamespace(MIN="min")
        def reduce(vote, *, op, group):
            self.assertEqual(op, "min")
            self.assertEqual(group, "cpu-group")
            votes.append(vote.value)
            vote.value = min(vote.value, *remote)
        dist.all_reduce = reduce
        torch.distributed = dist
        peer = ModuleType("vllm.distributed.device_communicators.custom_all_reduce")
        def can(*args):
            if isinstance(local, Exception):
                raise local
            return local
        peer._can_p2p = can
        ns = {"envs": SimpleNamespace(VLLM_FIREFLY_AR_BACKEND=preference), "logger": logging.getLogger("test"), "torch": torch}
        fn = source_function("vllm/distributed/device_communicators/firefly_allreduce.py", "_select_backend", ns, "FireflyAllReduce")
        obj = SimpleNamespace(_group="cpu-group" if group_present else None, world_size=4)
        with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist, peer.__name__: peer}):
            result = fn(obj, 0)
        return result, votes

    def test_partial_topology_uses_same_shm_backend_on_every_rank(self):
        capabilities = [True, False, True, True]
        for rank, local in enumerate(capabilities):
            with self.subTest(rank=rank):
                result, votes = self.select(local, capabilities[:rank] + capabilities[rank + 1:])
                self.assertEqual(result, "shm")
                self.assertEqual(votes, [int(local)])

    def test_complete_topology_uses_p2p_and_forced_p2p_still_checks(self):
        self.assertEqual(self.select(True, [True] * 3), ("p2p", [1]))
        self.assertEqual(self.select(True, [False, True, True], "p2p"), ("shm", [1]))

    def test_driver_exception_votes_false_instead_of_skipping_collective(self):
        self.assertEqual(self.select(RuntimeError("driver failure"), [True] * 3), ("shm", [0]))

    def test_explicit_shm_or_missing_group_cannot_enter_p2p(self):
        self.assertEqual(self.select(True, [True] * 3, "shm"), ("shm", []))
        self.assertEqual(self.select(True, [True] * 3, "p2p", False), ("shm", []))

    def test_one_rank_extension_failure_disables_every_rank(self):
        class Tensor:
            def __init__(self, values):
                self.value = values[0]
            def item(self):
                return self.value
        torch = ModuleType("torch")
        torch.__path__ = []
        torch.int32 = "int32"
        torch.tensor = lambda values, **kwargs: Tensor(values)
        dist = ModuleType("torch.distributed")
        dist.ReduceOp = SimpleNamespace(MIN="min")
        votes = []
        def reduce(vote, **kwargs):
            votes.append(vote.value)
            vote.value = 0  # one rank failed its extension load
        dist.all_reduce = reduce
        torch.distributed = dist
        for available in [True, False]:
            ns = {"torch": torch, "firefly_ar_world_ok": lambda n: True,
                  "firefly_ar_active": lambda: True,
                  "_load_cuda_mod": lambda: object() if available else None}
            fn = source_function("vllm/distributed/device_communicators/firefly_allreduce.py", "__init__", ns, "FireflyAllReduce")
            obj = SimpleNamespace()
            with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
                fn(obj, 0, 4, "cuda:0", "unused", "cpu-group")
            self.assertTrue(obj.disabled)
        self.assertEqual(votes, [1, 0])


if __name__ == "__main__":
    unittest.main()
