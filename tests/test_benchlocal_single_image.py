"""Verify the public single-image launch path without Docker or real credentials."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
class SingleImage(unittest.TestCase):
 def test_generated_full_recipe_matches_inputs(self):
  spec=importlib.util.spec_from_file_location('recipe',ROOT/'tools/generate_ultra_dockerfile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertEqual((ROOT/'docker/Dockerfile.ultra-full').read_text(),m.render())
  self.assertNotIn('local/sm75-benchlocal-',m.render())
 @unittest.skipUnless(shutil.which('bash'),'Bash required')
 def test_workers_use_one_image_and_no_gpu_or_host_mounts(self):
  with tempfile.TemporaryDirectory() as directory:
   d=Path(directory);log=d/'calls.jsonl';fake=d/'docker'
   fake.write_text('''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')
if args[:2]==['image','inspect']:print('1' if 'io.sm75.benchlocal.unified' in ' '.join(args) else 'sha256:fixture');sys.exit(0)
if args[:2] in [['container','inspect'],['network','inspect']]:sys.exit(1)
sys.exit(0)
''');fake.chmod(0o755)
   env={**os.environ,'PATH':str(d)+os.pathsep+os.environ['PATH'],'CALLS':str(log),'ULTRA_DATA_ROOT':str(d/'state'),'IMAGE':'fixture/ultra:beta2'}
   p=subprocess.run(['bash',str(ROOT/'docker/benchlocal.sh'),'start'],env=env,text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr)
   calls=[json.loads(x) for x in log.read_text().splitlines()];runs=[x for x in calls if x[:1]==['run']]
   self.assertEqual(len(runs),3)
   for args in runs:
    self.assertIn('fixture/ultra:beta2',args);self.assertNotIn('--gpus',args);self.assertNotIn('--volume',args)
    self.assertEqual(args[args.index('--runtime')+1],'runc');self.assertEqual(args[args.index('--memory')+1],'4g')
    self.assertIn('--read-only',args);self.assertIn('NVIDIA_VISIBLE_DEVICES=void',args)
   self.assertFalse(any(x[:1] in [['pull'],['build']] for x in calls))
   key=d/'state/benchlocal.env';self.assertRegex(key.read_text(),r'^SM75_BENCH_VERIFIER_KEY=[0-9a-f]{64}\n$')
   self.assertEqual(key.stat().st_mode & 0o777,0o600)
   self.assertNotIn(key.read_text().strip().split('=',1)[1],p.stdout+p.stderr)
 def test_runtime_payload_has_all_three_roles(self):
  root=ROOT/'ultra/source/benchlocal-verifiers'
  for role in ['bugfind-15','cli-40','hermesagent-20']:
   self.assertTrue((root/role/'server.mjs').is_file())
  lock=json.loads((root/'assets.lock.json').read_text())
  self.assertEqual(set(lock),{'go','rust','chrome','agent-browser'})
  for asset in lock.values():self.assertRegex(asset['sha256'],r'^[0-9a-f]{64}$')
if __name__=='__main__':unittest.main()
