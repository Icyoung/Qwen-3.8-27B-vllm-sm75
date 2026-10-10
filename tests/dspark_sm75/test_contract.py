import importlib.util
import pathlib
import unittest
from types import SimpleNamespace as N

HERE = pathlib.Path(__file__).resolve()
HELPER = HERE.parents[2] / "docker" / "speculative" / "sm75_dspark_compat.py"
spec = importlib.util.spec_from_file_location("dspark_compat_under_test", HELPER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config(anchor=True, sizes=None, maximum=8):
    return N(
        speculative_config=N(method="dspark", num_speculative_tokens=7,
            draft_model_config=N(hf_config=N(sample_from_anchor=anchor)),
            draft_tensor_parallel_size=4, kv_cache_dtype="auto",
            draft_sample_method="probabilistic", rejection_sample_method="standard",
            enable_adaptive_verification=False),
        model_config=N(dtype="float16"),
        parallel_config=N(tensor_parallel_size=4, pipeline_parallel_size=1,
                          decode_context_parallel_size=1),
        compilation_config=N(cudagraph_capture_sizes=[8] if sizes is None else sizes,
                             max_cudagraph_capture_size=maximum,
                             static_forward_context={"target": object()}),
        cache_config=N(cache_dtype="fp8_e4m3", layout="HND"))


class ContractTests(unittest.TestCase):
    def test_target8_draft7_without_target_mutation(self):
        c = config()
        d = m.draft_attention_config(c, 7)
        self.assertEqual(m.graph_capture_plan(c), (8, frozenset({8})))
        self.assertEqual(m.graph_capture_plan(d), (7, frozenset({7})))
        self.assertEqual(d.compilation_config.cudagraph_capture_sizes, [7])
        self.assertEqual(c.compilation_config.cudagraph_capture_sizes, [8])
        self.assertEqual(c.cache_config.cache_dtype, "fp8_e4m3")
        self.assertEqual(d.cache_config.cache_dtype, "auto")
        self.assertIs(d.compilation_config.static_forward_context,
                      c.compilation_config.static_forward_context)
        self.assertIs(d.model_config, c.model_config)

    def test_capture_request_counts_and_limit(self):
        self.assertEqual(m.remap_capture_sizes([1, 8, 9, 16, 17], 16, 8, 7),
                         ([7, 14], 14))
        self.assertEqual(m.remap_capture_sizes([8], 0, 8, 7), ([], 0))

    def test_bonus_anchor_keeps_eight_slots(self):
        c = config(anchor=False)
        d = m.draft_attention_config(c, 8)
        self.assertEqual(m.graph_capture_plan(d), (8, frozenset({8})))

    def test_k3_and_k5_are_explicit_runtime_shapes(self):
        for k in (3, 5):
            c = config(sizes=[k + 1], maximum=k + 1)
            c.speculative_config.num_speculative_tokens = k
            d = m.draft_attention_config(c, k)
            self.assertEqual(m.graph_capture_plan(d), (k, frozenset({k})))
            self.assertEqual(m.graph_capture_plan(c), (k + 1, frozenset({k + 1})))

    def test_real_runtime_width_must_match(self):
        with self.assertRaisesRegex(ValueError, "query mismatch"):
            m.draft_attention_config(config(), 8)

    def test_fail_closed_sampling_adaptive_and_tp(self):
        m.validate_dspark_config(config())
        for field, value in (("enable_adaptive_verification", True),
                             ("rejection_sample_method", "synthetic"),
                             ("draft_sample_method", "greedy"),
                             ("draft_tensor_parallel_size", 1),
                             ("kv_cache_dtype", "fp8")):
            c = config()
            setattr(c.speculative_config, field, value)
            with self.assertRaises(ValueError):
                m.validate_dspark_config(c)

    def test_unrelated_method_is_not_restricted(self):
        c = config()
        c.speculative_config.method = "dflash"
        c.speculative_config.enable_adaptive_verification = True
        m.validate_dspark_config(c)
        self.assertIs(m.draft_attention_config(c, 99), c)

    def test_group_names_not_prefixes(self):
        target = N(layer_names=["model.layers.1"], is_eagle_group=False)
        draft = N(layer_names=["custom.actual.layer.65"], is_eagle_group=False)
        self.assertEqual(m.annotate_draft_cache_groups(
            [N(kv_cache_groups=[target, draft])], {"custom.actual.layer.65"}), 1)
        self.assertFalse(target.is_eagle_group)
        self.assertTrue(draft.is_eagle_group)

    def test_group_preflight_is_atomic_for_missing_or_mixed(self):
        first = N(layer_names=["draft.a"], is_eagle_group=False)
        mixed = N(layer_names=["draft.a", "target.b"], is_eagle_group=False)
        for second in (N(kv_cache_groups=[]), N(kv_cache_groups=[mixed])):
            with self.assertRaises(ValueError):
                m.annotate_draft_cache_groups(
                    [N(kv_cache_groups=[first]), second], {"draft.a"})
            self.assertFalse(first.is_eagle_group)
        with self.assertRaises(ValueError):
            m.annotate_draft_cache_groups([], set())


if __name__ == "__main__":
    unittest.main()
