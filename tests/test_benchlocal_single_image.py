"""Verify the public single-image launch path without Docker or real credentials."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
class SingleImage(unittest.TestCase):
 def test_generated_full_recipe_matches_inputs(self):
  spec=importlib.util.spec_from_file_location('recipe',ROOT/'tools/generate_ultra_dockerfile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertEqual((ROOT/'docker/Dockerfile.ultra-full').read_text(),m.render())
  self.assertNotIn('local/sm75-benchlocal-',m.render())
 @unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux Bash required')
 def test_host_socket_path_is_bound_to_a_valid_controller_name(self):
  env={**os.environ,'CONTAINER_NAME':'fixture-ultra'}
  r=subprocess.run(['bash',str(ROOT/'docker/benchlocal.sh'),'path'],env=env,text=True,capture_output=True)
  self.assertEqual(r.returncode,0,r.stderr)
  import hashlib
  expected='/run/sm75-benchlocal/'+hashlib.sha256(b'fixture-ultra').hexdigest()[:24]+'/control'
  self.assertEqual(r.stdout.strip(),expected)
  for name in ['../escape','x/../../host','name;docker run anything','']:
   bad={**env,'CONTAINER_NAME':name}
   if not name:continue
   r=subprocess.run(['bash',str(ROOT/'docker/benchlocal.sh'),'path'],env=bad,text=True,capture_output=True)
   self.assertNotEqual(r.returncode,0)
 def test_execution_profile_has_the_documented_narrow_extension(self):
  root=ROOT/'ultra/source/benchlocal-verifiers'
  meta=json.loads((root/'browser-seccomp-source.json').read_text())
  profile=json.loads((root/'browser-seccomp.json').read_text())
  self.assertEqual(set(meta['additionalCalls']),{'clone','setns','unshare','chroot'})
  self.assertEqual(profile['defaultAction'],'SCMP_ACT_ERRNO')
  import hashlib
  self.assertEqual(hashlib.sha256((root/'browser-seccomp.json').read_bytes()).hexdigest(),meta['outputSha256'])
 def test_runtime_payload_has_all_three_roles(self):
  root=ROOT/'ultra/source/benchlocal-verifiers'
  for role in ['bugfind-15','cli-40','hermesagent-20']:
   self.assertTrue((root/role/'server.mjs').is_file())
  lock=json.loads((root/'assets.lock.json').read_text())
  self.assertEqual(set(lock),{'go','rust','chrome','agent-browser'})
  for asset in lock.values():self.assertRegex(asset['sha256'],r'^[0-9a-f]{64}$')
if __name__=='__main__':unittest.main()
