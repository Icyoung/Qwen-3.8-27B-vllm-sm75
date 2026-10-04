import importlib.util,os,sys,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

class Parent:
    def __init__(self,family='qwen4_exp_text',tp=8):
        self.vllm_config=NS(model_config=NS(hf_text_config=NS(model_type=family)),parallel_config=NS(tensor_parallel_size=tp))
        self.max_num_scheduled_tokens=4096;self.running=[];self.waiting=[];self.skipped_waiting=[]
        self.scheduler_config=NS(max_num_seqs=4);self.num_spec_tokens=3
    def schedule(self,throttle_prefills=False):
        if getattr(self,'fail',False):raise RuntimeError('Parent failure')
        return self.max_num_scheduled_tokens,throttle_prefills

module=types.ModuleType('vllm.v1.core.sched.async_scheduler');module.AsyncScheduler=Parent
with patch.dict(sys.modules,{'vllm.v1.core.sched.async_scheduler':module}):
    spec=importlib.util.spec_from_file_location('candidate',Path(__file__).resolve().parents[1]/'vllm/v1/core/sched/scheduler_sm75.py')
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)

class BudgetTests(unittest.TestCase):
    def create(self,**kw):
        with patch.dict(os.environ,{'VLLM_SM75_NEXT_MIXED_TOKEN_BUDGET':'1024'}):return candidate.SM75Scheduler(**kw)
    def test_single_prefill_keeps_full_budget(self):
        s=self.create();s.running=[NS(is_prefill_chunk=True)]
        self.assertEqual(s.schedule(),(4096,False))
    def test_pure_decode_keeps_full_budget(self):
        s=self.create();s.running=[NS(is_prefill_chunk=False)]*4
        self.assertEqual(s.schedule(),(4096,False))
    def test_mixed_caps_and_restores(self):
        s=self.create();s.running=[NS(is_prefill_chunk=False),NS(is_prefill_chunk=True)]
        self.assertEqual(s.schedule(throttle_prefills=True),(1024,True))
        self.assertEqual(s.max_num_scheduled_tokens,4096)
    def test_waiting_prompt_caps(self):
        s=self.create();s.running=[NS(is_prefill_chunk=False)];s.waiting=[object()]
        self.assertEqual(s.schedule()[0],1024)
    def test_parent_exception_restores(self):
        s=self.create();s.running=[NS(is_prefill_chunk=False),NS(is_prefill_chunk=True)];s.fail=True
        with self.assertRaises(RuntimeError):s.schedule()
        self.assertEqual(s.max_num_scheduled_tokens,4096)
    def test_other_model_does_not_change(self):
        s=self.create(family='qwen3_8');s.running=[NS(is_prefill_chunk=False),NS(is_prefill_chunk=True)]
        self.assertEqual(s.schedule()[0],4096)
    def test_other_tp_does_not_change(self):
        s=self.create(tp=4);s.running=[NS(is_prefill_chunk=False),NS(is_prefill_chunk=True)]
        self.assertEqual(s.schedule()[0],4096)
    def test_speculation_toggle_still_works(self):
        s=self.create();self.assertEqual(s.num_spec_tokens,3)
        s.set_spec_decode_enabled(False);self.assertEqual(s.num_spec_tokens,0)
        s.set_spec_decode_enabled(True);self.assertEqual(s.num_spec_tokens,3)

if __name__=='__main__':unittest.main(verbosity=2)
