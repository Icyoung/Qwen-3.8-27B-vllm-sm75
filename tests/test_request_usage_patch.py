"""Byte-exact vLLM 0.30 fixtures test the private active-usage patch without CUDA."""
import ast
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('patch_request_usage', ROOT/'docker/helpers/patch_request_usage.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


def original(kind):
    return gzip.decompress((ROOT/f'tests/fixtures/request_timing/v030-{kind}-serving.py.gz').read_bytes())


class StreamUsagePatch(unittest.TestCase):
    def test_actual_pinned_sources_are_strict_and_idempotent(self):
        for kind, (sha, _, _) in patcher.PATCHES.items():
            source = original(kind)
            self.assertEqual(hashlib.sha256(source).hexdigest(), sha)
            updated = patcher.patch_source(source, kind)
            self.assertNotEqual(updated, source)
            self.assertEqual(patcher.patch_source(updated, kind), updated)
            self.assertEqual(updated.decode().count('_sm75_stream_usage'), 1)
            ast.parse(updated)
            # The only source change is this private assignment after the exact
            # existing counter update. No parser, request or yield is replaced.
            self.assertEqual(updated.decode().replace(patcher.usage_block(kind), ''), source.decode())

    def test_unknown_modified_and_duplicate_sources_are_rejected(self):
        for kind in patcher.PATCHES:
            source = original(kind)
            updated = patcher.patch_source(source, kind)
            for bad in (source+b'\n# custom edit\n', updated+b'\n# custom edit\n',
                        updated.replace(b'_sm75_stream_usage', b'_other_usage'),
                        updated+patcher.usage_block(kind).encode()):
                with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                    patcher.patch_source(bad, kind)

    def test_real_generator_counter_and_hook_count_multi_token_batches_once(self):
        for kind in patcher.PATCHES:
            module = ast.parse(patcher.patch_source(original(kind), kind))
            generator = next(n for n in ast.walk(module)
                if isinstance(n, ast.AsyncFunctionDef) and n.name == kind+'_stream_generator')
            loop = next(n for n in ast.walk(generator)
                if isinstance(n, ast.For) and ast.unparse(n.target) == 'output')
            index = next(i for i, n in enumerate(loop.body)
                if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == 'previous_num_tokens[i]')
            block = ast.Module(body=loop.body[index:index+2], type_ignores=[])
            code = compile(ast.fix_missing_locations(block), '<actual v030 counter/hook>', 'exec')
            metadata = SimpleNamespace(final_usage_info=None)
            state = {'previous_num_tokens':[0], 'num_choices':1,'num_prompts':1,'i':0,
                'num_prompt_tokens':100 if kind == 'chat_completion' else [100],
                'request_metadata':metadata}
            for batch, total in ((4,4),(3,7),(0,7)):
                state['output'] = SimpleNamespace(token_ids=list(range(batch)))
                exec(code, state)
                self.assertEqual(metadata._sm75_stream_usage, {'prompt_tokens':100,'completion_tokens':total})
            self.assertIsNone(metadata.final_usage_info)
            for choice_count, prompts in ((2,1),(1,2)):
                if kind == 'chat_completion' and prompts == 2: continue
                state.update(num_choices=choice_count,num_prompts=prompts)
                metadata.__dict__.pop('_sm75_stream_usage')
                exec(code, state)
                self.assertNotIn('_sm75_stream_usage', metadata.__dict__)
                metadata.__dict__['_sm75_stream_usage'] = {}

    def test_package_prechecks_both_sources_and_repeat_install_does_not_add_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in patcher.PATCHES:
                file = root/f'entrypoints/openai/{kind}/serving.py'
                file.parent.mkdir(parents=True); file.write_bytes(original(kind))
            bad = root/'entrypoints/openai/completion/serving.py'
            bad.write_bytes(bad.read_bytes()+b'\n# mismatch')
            with self.assertRaises(RuntimeError): patcher.patch_package(root)
            good = root/'entrypoints/openai/chat_completion/serving.py'
            self.assertEqual(good.read_bytes(), original('chat_completion'))
            bad.write_bytes(original('completion'))
            self.assertEqual(patcher.patch_package(root), 2)
            self.assertEqual(patcher.patch_package(root), 0)
            self.assertEqual(list(root.rglob('.sm75-stream-usage-*')), [])

    def test_internal_metadata_is_not_added_to_pydantic_dump(self):
        try:
            from pydantic import BaseModel
        except ImportError:
            self.skipTest('Pydantic integration runs in the Linux vLLM fixture')
        class Metadata(BaseModel):
            request_id: str
            final_usage_info: dict | None = None
        metadata = Metadata(request_id='synthetic')
        metadata.__dict__['_sm75_stream_usage'] = {'prompt_tokens':100,'completion_tokens':4}
        self.assertEqual(metadata.model_dump(), {'request_id':'synthetic','final_usage_info':None})

    def test_full_and_fast_installers_bundle_the_guarded_patch(self):
        for recipe in ('Dockerfile','Dockerfile.fast'):
            self.assertIn('COPY docker/helpers/patch_request_usage.py', (ROOT/'docker'/recipe).read_text())
        installer = (ROOT/'docker/helpers/install_sm75_overlay.py').read_text(encoding='utf-8')
        self.assertIn('patch_request_usage(package_root)', installer)


if __name__ == '__main__': unittest.main()
