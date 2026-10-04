"""CPU-only lifecycle/config tests; no LMCache, Docker, CUDA or GPU required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lmcache_standard', ROOT / 'docker/helpers/lmcache-launch.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = {'LMCACHE_CHUNK_SIZE': '1568'}
ARGS = ['/models/test', '--tensor-parallel-size', '4', '--custom-unknown', 'keep']


class Child:
    def __init__(self, role, code=None):
        self.role, self.code = role, code
    def poll(self):
        return self.code


class LMCacheStandard(unittest.TestCase):
    def launcher(self, *, engine_code=0, fail=None, health=None):
        started, stopped = [], []
        def popen(command, **kwargs):
            role = 'engine' if command[0] == 'vllm' else ('pstate' if command[0] == 'bash' else 'cache')
            if role == fail:
                raise OSError('mock launch failed')
            child = Child(role, engine_code if role == 'engine' else None)
            started.append(child)
            self.assertTrue(kwargs['start_new_session'])
            return child
        def stop(child, **kwargs):
            stopped.append(child.role)
        launcher = m.Launcher(env=ENV, popen=popen, terminate=stop, health=health or (lambda _: True),
            probe=lambda _: {'runtimeFingerprint': 'audited-fixture'},
            port_check=lambda _: None, disk_check=lambda *a, **kw: None)
        return launcher, started, stopped

    def test_cpu_defaults_and_required_chunk(self):
        c = m.configuration(ARGS, ENV)
        self.assertEqual((c['cpuGiB'], c['diskPath'], c['port'], c['httpPort']), (8, '', 5555, 5556))
        for env in ({}, {'LMCACHE_CHUNK_SIZE': ''}, {'LMCACHE_CHUNK_SIZE': '0'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                m.configuration(ARGS, env)

    def test_explicit_conflicts_fail_before_probe_or_processes(self):
        for extra in (['--kv-transfer-config={}'], ['--kv-offloading-size','8'],
                      ['--speculative-config','{}'], ['--enable-sleep-mode'],
                      ['--enable-cumem-allocator'], ['--disable-hybrid-kv-cache-manager'],
                      ['--auto-sleep-idle-timeout','30']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                m.configuration(ARGS + extra, ENV)
        for env in ({'POWER_MODE':'sleep'}, {'CPU_KV_GIB':'8'}, {'KV_TRANSFER_CONFIG':'{}'},
                    {'AUTO_SLEEP_IDLE_TIMEOUT':'30'}, {'VLLM_AUTO_SLEEP_IDLE_TIMEOUT':'30'},
                    {'PYTORCH_ALLOC_CONF':'expandable_segments:True'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                m.configuration(ARGS, ENV | env)

    def test_ports_and_disk_root_fail_closed(self):
        for env in ({'LMCACHE_PORT':'5556'}, {'LMCACHE_HTTP_PORT':'8000'},
                    {'LMCACHE_PORT':'80'}, {'LMCACHE_DISK_PATH':'/'}, {'LMCACHE_DISK_PATH':'/a/../b'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                m.configuration(ARGS, ENV | env)

    def test_same_reviewed_mp_connector_and_storage_args_preserve_unknown_engine_args(self):
        args = ARGS + ['--no-enable-prefix-caching', '--mamba-cache-mode','all']
        p = m.plan(args, ENV, m.configuration(args, ENV), 'test')
        self.assertIn('--custom-unknown', p['engineArgs'])
        self.assertNotIn('--no-enable-prefix-caching', p['engineArgs'])
        self.assertEqual(m.arg_value(p['engineArgs'], '--mamba-cache-mode'), 'align')
        kv = json.loads(m.arg_value(p['engineArgs'], '--kv-transfer-config'))
        self.assertEqual(kv['kv_connector'], 'LMCacheMPConnector')
        self.assertFalse(kv['kv_connector_extra_config']['lmcache.mp.isolated_ipc'])
        self.assertIn('--separate-object-groups', p['serverArgs'])
        self.assertIn('--no-isolated-ipc', p['serverArgs'])
        self.assertNotIn('--l2-adapter', p['serverArgs'])

    def test_namespace_covers_model_layout_runtime_and_avoids_secret_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp); (model/'config.json').write_text('{}')
            args = [str(model), *ARGS[1:]]
            config = m.configuration(args, ENV)
            baseline = m.plan(args, ENV, config, 'runtime1')['namespace']
            for other in [m.plan(args, ENV, config, 'runtime2'),
                          m.plan(args + ['--kv-cache-dtype','auto'], ENV, config, 'runtime1'),
                          m.plan(args, ENV | {'VLLM_GDN_DECODE_KERNEL':'other'}, config, 'runtime1'),
                          m.plan(args, ENV, config | {'chunkSize':3136}, 'runtime1')]:
                self.assertNotEqual(other['namespace'], baseline)
            self.assertEqual(m.plan(args, ENV | {'VLLM_API_KEY':'synthetic-secret'}, config, 'runtime1')['namespace'], baseline)
            (model/'config.json').write_text('{"different":"layout"}')
            self.assertNotEqual(m.plan(args, ENV, config, 'runtime1')['namespace'], baseline)
            self.assertTrue(baseline.startswith('standard-v1-'))

    def test_persistent_cache_requires_local_model_identity(self):
        config = m.configuration(ARGS, ENV | {'LMCACHE_DISK_PATH':'/lmcache'})
        with self.assertRaisesRegex(ValueError, 'local model'):
            m.plan(ARGS, ENV, config)

    def test_normal_exit_reaps_model_before_cache_and_preserves_disk(self):
        launcher, started, stopped = self.launcher()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/'model'; model.mkdir(); (model/'config.json').write_text('{}')
            disk = root/'cache'; disk.mkdir(); sentinel = disk/'user-cache'; sentinel.write_text('keep')
            args = [str(model), *ARGS[1:]]
            config = m.configuration(args, ENV) | {'diskPath':str(disk), 'minFreeDiskGiB':0}
            launcher.disk_check = m.check_disk
            with mock.patch.object(m, 'configuration', return_value=config):
                self.assertEqual(launcher.run(args), 0)
            self.assertEqual(sentinel.read_text(), 'keep')
            self.assertTrue(list(disk.glob('standard-v1-*')))
        self.assertEqual([c.role for c in started], ['cache','pstate','engine'])
        self.assertEqual(stopped, ['engine','pstate','cache'])

    def test_engine_spawn_failure_cleans_every_started_service(self):
        launcher, _, stopped = self.launcher(fail='engine')
        with self.assertRaisesRegex(OSError, 'mock launch'):
            launcher.run(ARGS)
        self.assertEqual(stopped, ['pstate','cache'])

    def test_late_healthy_response_does_not_override_cancel(self):
        launcher, started, stopped = self.launcher()
        def health(_):
            launcher.cancel(); return True
        launcher.health = health
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            launcher.run(ARGS)
        self.assertEqual([c.role for c in started], ['cache'])
        self.assertEqual(stopped, ['cache'])

    def test_running_cancel_releases_engine_first(self):
        launcher, _, stopped = self.launcher(engine_code=None)
        launcher.sleep = lambda _: launcher.cancel()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            launcher.run(ARGS)
        self.assertEqual(stopped, ['engine','pstate','cache'])

    def test_readiness_failure_stops_cache_without_engine(self):
        launcher, started, stopped = self.launcher(health=lambda _: False)
        launcher.timeout = 0
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            launcher.run(ARGS)
        self.assertEqual([c.role for c in started], ['cache'])
        self.assertEqual(stopped, ['cache'])

    def test_probe_failure_never_starts_cache_or_engine(self):
        launcher, started, stopped = self.launcher()
        def reject(_): raise RuntimeError('bad ABI')
        launcher.probe = reject
        with self.assertRaisesRegex(RuntimeError, 'bad ABI'): launcher.run(ARGS)
        self.assertEqual(started, []); self.assertEqual(stopped, [])

    def test_cleanup_attempts_cache_even_if_engine_cleanup_fails(self):
        launcher, _, stopped = self.launcher()
        def stop(child, **kwargs):
            stopped.append(child.role)
            if child.role == 'engine': raise OSError('stop failed')
        launcher.terminate = stop
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'): launcher.run(ARGS)
        self.assertEqual(stopped, ['engine','pstate','cache'])

    def test_real_probe_cancellation_reaps_child_without_gpu(self):
        child = mock.Mock()
        launcher = m.Launcher(env=ENV, popen=lambda *a, **kw: child, terminate=mock.Mock())
        def communicate(**kw):
            launcher.cancel(); raise subprocess.TimeoutExpired('mock probe', .2)
        child.communicate.side_effect = communicate
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            launcher.run_probe({'serverArgs':['-m','unused','--chunk-size','1568']})
        launcher.terminate.assert_called_once_with(child)
        self.assertEqual(launcher.children, {})

    def test_package_keeps_same_installer_probe_and_opt_in_default(self):
        for name in ('Dockerfile','Dockerfile.fast','Dockerfile.lmcache'):
            text = (ROOT/'docker'/name).read_text(encoding='utf-8')
            for value in ('COPY ultra/source/lmcache', 'COPY ultra/source/console/lmcache_probe.py',
                          'COPY docker/helpers/lmcache-launch.py', '--install --build-no-gpu'):
                self.assertIn(value, text)
            if name != 'Dockerfile.lmcache': self.assertIn('ARG INSTALL_LMCACHE=0', text)


if __name__ == '__main__': unittest.main()
