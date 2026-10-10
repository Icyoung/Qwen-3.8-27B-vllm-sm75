"""Exercise the actual patched FA2 control flow with CPU-only fake backends."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace as N
from unittest import TestCase, main
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
hs = importlib.util.spec_from_file_location("sm75_dspark_compat", ROOT / "docker/speculative/sm75_dspark_compat.py")
helper = importlib.util.module_from_spec(hs)
hs.loader.exec_module(helper)



class FA2Tests(TestCase):
    def adapter(self):
        modules = {"sm75_dspark_compat": helper}
        def module(name):
            if name not in modules:
                modules[name] = ModuleType(name)
                modules[name].__path__ = []
                if "." in name:
                    parent, leaf = name.rsplit(".", 1)
                    setattr(module(parent), leaf, modules[name])
            return modules[name]
        torch = module("torch")
        torch.float16, torch.float8_e4m3fn, torch.uint8 = "fp16", "fp8", "uint8"
        torch.empty = lambda n, **kw: N(size=n)
        torch.int32 = "int32"
        fi = module("vllm.v1.attention.backends.flashinfer")
        class Builder:
            @classmethod
            def get_cudagraph_support(cls, config, spec): return "native"
            def build(self, prefix, metadata, fast_build=False):
                return self._get_prefill_wrapper(metadata.causal)
            def _get_prefill_wrapper(self, causal=True): return "native"
            def _get_workspace_buffer(self): return "workspace"
        fi.FlashInferMetadataBuilder = Builder
        fi.get_flashinfer_layout_string = lambda layout: layout
        fi.BatchPrefillWithPagedKVCacheWrapper = lambda *a, **kw: N(kind="graph", **kw)
        module("vllm.v1.attention.backend").AttentionCGSupport = N(UNIFORM_BATCH="uniform")
        class AttentionSpec: pass
        class UniformTypeKVCacheSpecs: pass
        kv = module("vllm.v1.kv_cache_interface")
        kv.AttentionSpec, kv.UniformTypeKVCacheSpecs = AttentionSpec, UniformTypeKVCacheSpecs
        module("vllm.platforms").current_platform = N(is_device_capability=lambda n: n == 75)
        module("vllm.envs").VLLM_BATCH_INVARIANT = False
        module("vllm.logger").init_logger = lambda name: N(info=lambda *a: None, info_once=lambda *a: None)
        source = (ROOT / "docker/speculative/sm75_fa2_graph.py").read_text(encoding="utf-8")
        namespace = {}
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {"SM75_FA2_SMALLQ_GRAPH": "1"}):
            exec(compile(source, "patched_fa2", "exec"), namespace)
            namespace["install"]()
        config = N(speculative_config=N(method="dspark", num_speculative_tokens=7,
                   kv_cache_dtype="auto", draft_model_config=N(hf_config=N(sample_from_anchor=True))),
                   compilation_config=N(cudagraph_capture_sizes=[8], max_cudagraph_capture_size=8),
                   cache_config=N(cache_dtype="fp8"), model_config=N(dtype="fp16"),
                   parallel_config=N(decode_context_parallel_size=1))
        self.modules = modules
        return Builder, config

    def builder(self, cls, config):
        b = cls()
        b.vllm_config = config
        b.use_dcp = b.has_sinks = b.is_kvcache_nvfp4 = False
        b.q_data_type_prefill = b.kv_cache_dtype = "fp16"
        b.head_dim, b.num_qo_heads, b.num_kv_heads = 128, 8, 2
        b.enable_cuda_graph = True
        b.prefill_fixed_split_size, b.disable_split_kv = -1, False
        b.model_config = N(max_model_len=4096)
        b.page_size, b.device, b.kv_cache_layout = 32, "fake-device", "HND"
        return b

    def test_target8_draft7_actual_adapter_and_causal_keys(self):
        cls, c = self.adapter()
        d = helper.draft_attention_config(c, 7)
        with patch.dict(sys.modules, self.modules):
            for config, q in ((c, 8), (d, 7)):
                b = self.builder(cls, config)
                meta = N(num_reqs=1, max_query_len=q, num_actual_tokens=q, causal=False)
                first = b.build(0, meta)
                self.assertEqual(first.kind, "graph")
                self.assertEqual(first.qo_indptr_buf.size, 2)
                self.assertIsNone(b._sm75_smallq_key)
                meta.causal = True
                self.assertIsNot(first, b.build(0, meta))

    def test_nonuniform_and_multi_request_keep_native_fallback(self):
        cls, c = self.adapter()
        d = helper.draft_attention_config(c, 7)
        with patch.dict(sys.modules, self.modules):
            for reqs, total, prefix in ((2, 14, 0), (2, 8, 0), (1, 7, 1)):
                b = self.builder(cls, d)
                meta = N(num_reqs=reqs, max_query_len=7, num_actual_tokens=total, causal=False)
                self.assertEqual(b.build(prefix, meta), "native")
                self.assertIsNone(b._sm75_smallq_key)

if __name__ == "__main__":
    main()
