import runpy, unittest
from pathlib import Path
from collections import namedtuple
api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'docker/helpers/cpu-kv-check.py'))
G=2**30
Space=namedtuple('Space','total used free')
def usage(total,free):return lambda _: Space(total,total-free,free)
class Check(unittest.TestCase):
 def test_native_shared_total(self):
  import json
  args=['/model','--tensor-parallel-size','4','--kv-transfer-config',json.dumps({'kv_connector':'OffloadingConnector','kv_connector_extra_config':{'cpu_bytes_to_use':16*G}})]
  self.assertEqual(api['native_cpu_bytes'](args),16*G)
  with self.assertRaisesRegex(ValueError,'32g'):api['check'](args,{},usage(16*G,16*G-4096))
  self.assertEqual(api['check'](args,{},usage(32*G,32*G-4096))['requiredBytes'],16*G)
 def test_env_priority(self):
  self.assertEqual(api['native_cpu_bytes'](['/model','--kv-offloading-size=8'],{'CPU_KV_GIB':'16'}),16*G)
  self.assertEqual(api['native_cpu_bytes'](['/model'],{'CPU_KV_GIB':'16','KV_TRANSFER_CONFIG':''}),0)
  self.assertEqual(api['native_cpu_bytes'](['/model'],{'CPU_KV_GIB':'0'}),0)
 def test_custom_lmcache_skip(self):
  for args,env in [(['/model'],{}),(['/model','--kv-transfer-config={"kv_connector":"Custom"}'],{}),(['/model','--kv-offloading-size','16','--kv-offloading-backend','lmcache'],{}),(['/model'],{'LMCACHE_ENABLED':'1','CPU_KV_GIB':'16'})]:
   self.assertIsNone(api['check'](args,env,lambda _:self.fail('probe')))
 def test_inline_and_default(self):
  self.assertEqual(api['native_cpu_bytes'](['/model','--kv-offloading-size=16']),16*G)
  self.assertEqual(api['native_cpu_bytes'](['/model','--kv-transfer-config={}']),8*G)
 def test_env_reaches_actual_args(self):
  args=api['apply_cpu_environment'](['/model','--kv-offloading-size=8'],{'CPU_KV_GIB':'16'})
  self.assertEqual(api['native_cpu_bytes'](args),16*G)
  self.assertNotIn('--kv-offloading-size=8',args)
  self.assertEqual(api['apply_cpu_environment'](args,{'CPU_KV_GIB':'0'}),['/model'])
  self.assertEqual(api['apply_cpu_environment'](args,{'CPU_KV_GIB':'16','KV_TRANSFER_CONFIG':''}),['/model'])
 def test_launcher_precedes_gpu_start(self):
  text=(Path(__file__).resolve().parents[1]/'docker/helpers/runtime-entrypoint.py').read_text()
  self.assertLess(text.index("with_name('cpu-kv-check.py')"),text.index("if env.get('POWER_MODE')=='pstate'"))
if __name__=='__main__':unittest.main()
