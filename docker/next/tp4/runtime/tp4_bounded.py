"""Test-only fix: release stale prefetch CPU tensors after each quantized module."""
import runpy, contextlib, os, gc, ctypes
_trim=ctypes.CDLL("libc.so.6").malloc_trim
import vllm.model_executor.model_loader.utils as loader
from vllm.model_executor.offloader import get_offloader
from vllm.model_executor.offloader.prefetch import PrefetchOffloader
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
original_context=loader.device_loading_context
@contextlib.contextmanager
def bounded_context(module, target_device):
    with original_context(module,target_device) as result:
        yield result
    offloader=get_offloader()
    if isinstance(offloader,PrefetchOffloader):
        # Repacking replaces Parameters. The prefetch pool otherwise retains
        # all old CPU weights until global post_init, doubling host memory.
        for owner in offloader.module_offloaders:
            owner.sync_cpu_storage()
        _trim(0)
loader.device_loading_context=bounded_context
original_post=PrefetchOffloader.post_init
def post_init(self):
    original_post(self)
    for owner in self.module_offloaders:
        for module in owner.module.modules():
            method=getattr(module,'quant_method',None)
            if isinstance(method,AutoGPTQMoEMethod):
                # Rebind quant config after CPU params point to static CUDA buffers.
                method._setup_kernel(module)
    if os.environ.get('TP4_PIN_HOT','0') == '1':
        pinned_bytes=0
        for owner in self.module_offloaders:
            for param in owner._param_offloaders.values():
                param._cpu_storage=param._cpu_storage.pin_memory()
                assert param._cpu_storage.is_pinned()
                pinned_bytes += param._cpu_storage.numel()*param._cpu_storage.element_size()
        print('TP4_PINNED_EXPERT_BYTES',os.getpid(),pinned_bytes,flush=True)
    gc.collect()
    _trim(0)
    print('TP4 RSS after trim',open('/proc/self/status').read().split('VmRSS:')[1].splitlines()[0],flush=True)
    print('TP4 bounded CPU storage and kernel rebind complete',os.getpid(),flush=True)
PrefetchOffloader.post_init=post_init
if __name__=='__main__':runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
