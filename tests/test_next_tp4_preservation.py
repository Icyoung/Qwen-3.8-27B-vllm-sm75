import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PAYLOAD=ROOT/'docker/next/tp4'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def config(tp=4,model='qwen4_exp_text',spec=None):
    return NS(model_config=NS(hf_text_config=NS(model_type=model,num_experts=512,hidden_size=2560),dtype='float16'),parallel_config=NS(tensor_parallel_size=tp,pipeline_parallel_size=1,enable_expert_parallel=True),speculative_config=spec)

class TP4Preservation(unittest.TestCase):
    def test_other_modes_never_import_tp4_hooks(self):
        m=module(PAYLOAD/'sm75_next_tp4.py','tp4_noop')
        with patch.dict(os.environ,{},clear=True),patch.object(m.importlib,'import_module') as imports:
            self.assertFalse(m.install(None));imports.assert_not_called()
    def test_wrong_model_tp_or_speculation_rejected_before_hooking(self):
        m=module(PAYLOAD/'sm75_next_tp4.py','tp4_guard')
        with patch.dict(os.environ,{'TP4_NEXT_RUNTIME':'1','SM75_PLE_STORE':'/ple','SM75_PLE_MODEL':'/model'},clear=True),patch.object(m.importlib,'import_module') as imports:
            for c in [config(tp=8),config(model='qwen3_5'),config(spec=object())]:
                with self.assertRaises(ValueError):m.install(c)
            imports.assert_not_called()
    def fixture(self,site):
        manifest=json.loads((PAYLOAD/'manifest.json').read_text())
        for name,meta in manifest['flashinfer_files'].items():
            if meta['kind']=='append':
                p=site/'flashinfer'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes((PAYLOAD/'flashinfer'/name).read_bytes()[:meta['base_bytes']])
        return manifest
    def test_installer_is_complete_idempotent_and_detects_unknown_source(self):
        m=module(PAYLOAD/'install.py','tp4_installer')
        with tempfile.TemporaryDirectory() as td:
            site=Path(td);manifest=self.fixture(site)
            changes,_=m.prepare(PAYLOAD,site)
            self.assertEqual(len(changes),23)
            for data,p in changes:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            again,_=m.prepare(PAYLOAD,site);self.assertEqual(len(again),23)
            target=site/'flashinfer/jit/comm.py';target.write_text('unreviewed changes')
            with self.assertRaisesRegex(ValueError,'Unreviewed FlashInfer'):m.prepare(PAYLOAD,site)
    def test_failed_preflight_does_not_install_partial_hooks(self):
        m=module(PAYLOAD/'install.py','tp4_preflight')
        with tempfile.TemporaryDirectory() as td:
            site=Path(td);self.fixture(site)
            (site/'flashinfer/jit/comm.py').write_text('unknown base')
            with self.assertRaises(ValueError):m.prepare(PAYLOAD,site)
            self.assertFalse((site/'sm75_next_tp4.py').exists())
    def test_cli_profile_keeps_requested_defaults_and_model_binding(self):
        m=module(ROOT/'docker/helpers/runtime-entrypoint.py','tp4_profile')
        m.profile_root=lambda:ROOT/'ultra/source/profiles'
        p=m.resolve(['--profile','flash-next-tp4-32k'],{'MODEL':'/models/custom-next'})
        self.assertEqual(p['env']['VLLM_SM75_QWEN38_HC_GEMV'],'0')
        self.assertEqual(p['env']['VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC'],'1')
        self.assertEqual(p['env']['SM75_PLE_MODEL'],'/models/custom-next')
        self.assertEqual(p['args'][0],'/models/custom-next')
        self.assertEqual(p['args'][p['args'].index('--kv-cache-memory-bytes')+1],'536870912')
        self.assertNotIn('--speculative-config',p['args'])
        override=m.resolve(['--profile','flash-next-tp4-32k'],{'VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC':'0'})
        self.assertEqual(override['env']['VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC'],'0')
        for identifier in ['flash-next-tp8-256k-mtp3','qwen38-27b-fp8-dspark5']:
            self.assertNotEqual(m.resolve(['--profile',identifier],{})['env'].get('TP4_NEXT_RUNTIME'),'1')

if __name__=='__main__':unittest.main()
