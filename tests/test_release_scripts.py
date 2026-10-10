"""Exercise public command construction without Docker, GPUs or real credentials."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash') or ('C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else None)


@unittest.skipUnless(BASH and Path(BASH).exists(), 'Bash required')
class ReleaseScripts(unittest.TestCase):
    def invoke(self, script, arguments=(), **settings):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # Bash printf emits one argument per line; no shell eval is used.
            fake = directory / 'docker'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$COMMAND_LOG"\n', encoding='utf-8')
            fake.chmod(0o755)
            logfile = directory / 'calls'
            env = {**os.environ, 'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                   'COMMAND_LOG': logfile.as_posix(), 'SOURCE_REVISION': 'test-source', 'BENCHLOCAL':'0',
                   'VLLM_API_KEY': 'synthetic-test-key', 'VLLM_SM75_CACHE_ROOT': '/tmp/sm75-test-cache',
                   **settings}
            env = {key: value for key, value in env.items() if value is not None}
            result = subprocess.run([BASH, str(ROOT / script), *arguments], env=env, cwd=ROOT,
                                    text=True, capture_output=True)
            calls = logfile.read_text().splitlines() if logfile.exists() else []
            return result, calls

    def test_standard_and_ultra_build_tags_share_version(self):
        version = (ROOT / 'docker/VERSION').read_text().strip()
        ultra_version = version.replace('-beta','-ultra-beta',1) if '-beta' in version else version + '-ultra'
        for edition, expected in [('standard', version), ('ultra', ultra_version)]:
            result, args = self.invoke('docker/build.sh', EDITION=edition, BUILD_MODE='full')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('vllm-sm75:v' + expected, args)

    def test_standard_mode_reuses_unified_image_and_keeps_shm_override(self):
        result,args=self.invoke('docker/run.sh',EDITION='standard',POWER_MODE='sleep',CPU_KV_GIB='0',SHM_SIZE='48g')
        self.assertEqual(result.returncode,0,result.stderr)
        version=(ROOT/'docker/VERSION').read_text().strip().replace('-beta','-ultra-beta',1)
        self.assertIn('vllm-sm75:v'+version,args)
        self.assertIn('SM75_EDITION=standard',args)
        self.assertEqual(args[args.index('--shm-size')+1],'48g')

    def test_public_shell_launchers_use_linux_line_endings(self):
        for path in (ROOT / 'docker').rglob('*.sh'):
            self.assertNotIn(b'\r\n', path.read_bytes(), str(path))

    def test_fast_and_ui_are_explicit_modes(self):
        result, args = self.invoke('docker/build.sh', EDITION='standard', BUILD_MODE='fast')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(a.endswith('/docker/Dockerfile.fast') for a in args))
        result, args = self.invoke('docker/build.sh', EDITION='ultra', BUILD_MODE='ui', RUNTIME_IMAGE='audited-runtime')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('BASE_IMAGE=audited-runtime', args)
        self.assertIn('--memory', args)
        self.assertEqual(args[args.index('--network') + 1], 'default')

    def test_lmcache_build_opt_in_reaches_all_recipes(self):
        for edition, mode in [('standard','full'),('standard','fast'),('ultra','full'),('ultra','ui')]:
            result, args = self.invoke('docker/build.sh', EDITION=edition, BUILD_MODE=mode,
                                      RUNTIME_IMAGE='audited-runtime', INSTALL_LMCACHE='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('INSTALL_LMCACHE=1', args)

    def test_context_template_is_forwarded_and_zero_cpu_disables_connector(self):
        override = json.dumps({'dtype': 'float16', 'text_config': {'rope_parameters':
            {'rope_type': 'yarn', 'factor': 4, 'original_max_position_embeddings': 262144},
            'max_position_embeddings': 1048576}})
        result, args = self.invoke('docker/run.sh', POWER_MODE='sleep',
            AUTO_SLEEP_IDLE_TIMEOUT='0', CPU_KV_GIB='0', GPU_KV_BYTES='4294967296',
            MAX_MODEL_LEN='1048576', MAX_NUM_SEQS='1', MAX_NUM_BATCHED_TOKENS='4096',
            HF_OVERRIDES=override)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--kv-transfer-config', args)
        for flag, value in [('--max-model-len','1048576'), ('--max-num-seqs','1'),
                            ('--max-num-batched-tokens','4096'),
                            ('--kv-cache-memory-bytes','4294967296'), ('--hf-overrides',override)]:
            self.assertEqual(args[args.index(flag)+1], value)
        result, args = self.invoke('docker/run.sh', POWER_MODE='sleep', CPU_KV_GIB='2')
        self.assertEqual(result.returncode, 0, result.stderr)
        kv = json.loads(args[args.index('--kv-transfer-config')+1])
        self.assertEqual(kv['kv_connector_extra_config']['cpu_bytes_to_use'], 2 * 1024**3)

    def test_decimal_cpu_gib_rounds_to_panel_safe_bytes_without_host_python(self):
        for value, expected in [('10.25',11005853696),('.25',268435456),('10.',10737418240),
                                ('0.1',107374182),('0.0000000005',1),
                                ('8388607.999999999',9007199254740991)]:
            with self.subTest(value=value):
                result, args = self.invoke('docker/run.sh', POWER_MODE='sleep',
                    AUTO_SLEEP_IDLE_TIMEOUT='0', CPU_KV_GIB=value)
                self.assertEqual(result.returncode,0,result.stderr)
                kv=json.loads(args[args.index('--kv-transfer-config')+1])
                self.assertEqual(kv['kv_connector_extra_config']['cpu_bytes_to_use'],expected)
        for value in ('0','0.0','.0','00.000'):
            result,args=self.invoke('docker/run.sh',POWER_MODE='sleep',CPU_KV_GIB=value)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('--kv-transfer-config',args)

    def test_invalid_and_out_of_range_decimal_cpu_gib_cannot_launch(self):
        for value in ('','-1','+1','1e3','0x10','NaN','Infinity','10.25GiB','1,25','1.2.3',
                      ' 10.25 ','0.0000000001','8388608','999999999999999999999999999999999999999999'):
            with self.subTest(value=value):
                result,args=self.invoke('docker/run.sh',POWER_MODE='sleep',CPU_KV_GIB=value)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('CPU_KV_GIB',result.stderr)
                self.assertEqual(args,[])

    def test_lmcache_rejects_decimal_cpu_offload_and_accepts_decimal_zero(self):
        result,args=self.invoke('docker/run.sh',LMCACHE_ENABLED='1',LMCACHE_CHUNK_SIZE='1568',
            CPU_KV_GIB='10.25')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('LMCache conflicts',result.stderr)
        self.assertEqual(args,[])
        result,args=self.invoke('docker/run.sh',LMCACHE_ENABLED='1',LMCACHE_CHUNK_SIZE='1568',
            CPU_KV_GIB='0.0',PSTATE_NVAPI_LIB='/bin/bash')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('/opt/vllm-sm75/lmcache-entrypoint.sh',args)
        self.assertNotIn('--kv-transfer-config',args)

    def test_nccl_default_and_override_reach_launchers(self):
        for script, edition in [('docker/run.sh', 'standard'),
                                ('docker/run.sh', 'ultra'),
                                ('docker/helpers/run-ultra.sh', 'ultra')]:
            for value, expected in [(None, 'SYS'), ('PIX', 'PIX'), ('', '')]:
                with self.subTest(script=script, edition=edition, value=value):
                    result, args = self.invoke(script, EDITION=edition,
                        POWER_MODE='sleep', AUTO_SLEEP_IDLE_TIMEOUT='0',
                        ULTRA_DATA_ROOT='/tmp/sm75-test-ultra', MODEL_ROOT='/tmp',
                        NCCL_P2P_LEVEL=value)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(args.count('NCCL_P2P_LEVEL=' + expected), 1)


    def test_core_standard_environment_and_last_cli_override(self):
        spec = '{"method":"mtp","num_speculative_tokens":3}'
        attention = '{"backend":"FLASH_ATTN"}'
        result, args = self.invoke('docker/run.sh', arguments=['--tensor-parallel-size', '1'],
            EDITION='standard', POWER_MODE='sleep', AUTO_SLEEP_IDLE_TIMEOUT='0',
            TENSOR_PARALLEL_SIZE='2', KV_CACHE_DTYPE='auto', BLOCK_SIZE='16', DTYPE='float16',
            TOOL_CALL_PARSER='qwen3_coder', REASONING_PARSER='qwen3', ENABLE_AUTO_TOOL_CHOICE='1',
            ATTENTION_CONFIG=attention, SPECULATIVE_CONFIG=spec, COMPILATION_CONFIG='',
            ENABLE_PREFIX_CACHING='false', ASYNC_SCHEDULING='0', GPU_KV_BYTES='0',
            KV_TRANSFER_CONFIG='', VLLM_USE_NCCL_SYMM_MEM='1', GPUS='device=0,1')
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag, expected in [('--tensor-parallel-size','2'),('--kv-cache-dtype','auto'),
                               ('--block-size','16'),('--tool-call-parser','qwen3_coder'),
                               ('--reasoning-parser','qwen3'),('--speculative-config',spec),
                               ('--attention-config',attention),('--gpus','device=0,1')]:
            self.assertEqual(args[args.index(flag)+1], expected)
        self.assertEqual(args[-2:], ['--tensor-parallel-size','1'])
        for flag in ['--no-enable-prefix-caching','--no-async-scheduling','--enable-auto-tool-choice',
                     'VLLM_USE_NCCL_SYMM_MEM=1','VLLM_API_KEY']:
            self.assertIn(flag,args)
        for absent in ['--kv-transfer-config','--compilation-config','--api-key','synthetic-test-key']:
            self.assertNotIn(absent,args)

    def test_ultra_passes_explicit_core_environment_without_putting_secrets_in_argv(self):
        result, args = self.invoke('docker/run.sh', EDITION='ultra',
            ULTRA_DATA_ROOT='/tmp/sm75-test-ultra', MODEL_ROOT='/tmp',
            TENSOR_PARALLEL_SIZE='2', CPU_KV_GIB='0', SPECULATIVE_CONFIG='',
            POWER_MODE='sleep', AUTO_SLEEP_IDLE_TIMEOUT='30', TOOL_CALL_PARSER='qwen3_coder')
        self.assertEqual(result.returncode, 0, result.stderr)
        for key in ['TENSOR_PARALLEL_SIZE','CPU_KV_GIB','SPECULATIVE_CONFIG','POWER_MODE',
                    'AUTO_SLEEP_IDLE_TIMEOUT','TOOL_CALL_PARSER','VLLM_API_KEY']:
            self.assertIn(key,args)
        self.assertNotIn('synthetic-test-key',args)

    def test_lmcache_standard_cpu_default_and_dedicated_disk_mount(self):
        result, args = self.invoke('docker/run.sh', POWER_MODE='pstate', PSTATE_NVAPI_LIB='/bin/bash',
            LMCACHE_ENABLED='1', LMCACHE_CHUNK_SIZE='1568', LMCACHE_CPU_GIB='8',
            LMCACHE_DISK_ROOT='/tmp/sm75-test-lmcache', CPU_KV_GIB=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/opt/vllm-sm75/lmcache-entrypoint.sh', args)
        self.assertNotIn('--kv-transfer-config', args)
        self.assertIn('/tmp/sm75-test-lmcache:/lmcache', args)
        for key in ('LMCACHE_ENABLED','LMCACHE_CHUNK_SIZE','LMCACHE_CPU_GIB','LMCACHE_DISK_PATH'):
            self.assertIn(key, args)
        self.assertNotIn('5555:5555', args); self.assertNotIn('5556:5556', args)

    def test_lmcache_explicit_conflicts_do_not_call_docker(self):
        for settings in ({'CPU_KV_GIB':'8'},{'KV_TRANSFER_CONFIG':'{}'},
                         {'POWER_MODE':'sleep'},{'VARIANT':'mtp'},{'SPECULATIVE_CONFIG':'{}'},
                         {'AUTO_SLEEP_IDLE_TIMEOUT':'30'}):
            result, args = self.invoke('docker/run.sh', LMCACHE_ENABLED='1',
                LMCACHE_CHUNK_SIZE='1568', **settings)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(args, [])

    def test_lmcache_ultra_environment_and_mount(self):
        result, args = self.invoke('docker/run.sh', EDITION='ultra',
            ULTRA_DATA_ROOT='/tmp/sm75-test-ultra', MODEL_ROOT='/tmp',
            LMCACHE_ENABLED='1', LMCACHE_CHUNK_SIZE='1568', LMCACHE_DISK_ROOT='/tmp/sm75-test-lmcache')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('LMCACHE_ENABLED', args); self.assertIn('LMCACHE_CHUNK_SIZE', args)
        self.assertIn('LMCACHE_DISK_PATH', args)
        self.assertIn('/tmp/sm75-test-lmcache:/lmcache', args)

    def test_invalid_mode_cannot_call_docker(self):
        result, args = self.invoke('docker/build.sh', EDITION='unknown')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(args, [])

    def test_standard_runtime_keeps_optimization_contract(self):
        result, args = self.invoke('docker/run.sh', EDITION='standard', POWER_MODE='sleep',
                                  VARIANT='base', FORMAT='fp8', IMAGE='test-image',
                                  VLLM_FIREFLY=None, VLLM_FIREFLY_AR=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ['test-image', 'VLLM_FIREFLY=1', 'VLLM_FIREFLY_AR=auto',
                      '--async-scheduling', 'flashqla_sm75', '--enable-prefix-caching']:
            self.assertIn(value, args)
        self.assertEqual(args[args.index('--auto-sleep-idle-timeout') + 1], '30')
        kv = json.loads(args[args.index('--kv-transfer-config') + 1])
        self.assertEqual(kv['kv_connector_extra_config']['cpu_bytes_to_use'], 8589934592)

    def test_marlin_reduction_setting_reaches_container(self):
        for value in ('0', '1'):
            result, args = self.invoke('docker/run.sh', POWER_MODE='sleep',
                AUTO_SLEEP_IDLE_TIMEOUT='0', VLLM_MARLIN_USE_ATOMIC_ADD=value)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('VLLM_MARLIN_USE_ATOMIC_ADD=' + value, args)

    def test_ultra_dispatch_does_not_require_web_token_or_stop_services(self):
        result, args = self.invoke('docker/run.sh', EDITION='ultra', ULTRA_DATA_ROOT='/tmp/sm75-test-ultra',
                                  MODEL_ROOT='/tmp', VLLM_API_KEY='', IMAGE='test-ultra')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('test-ultra', args)
        self.assertIn('SM75_CONSOLE_ROOT=/data', args)
        self.assertIn('no-new-privileges', args)
        self.assertNotIn('stop', args)
        self.assertNotIn('--privileged', args)


if __name__ == '__main__':
    unittest.main()
