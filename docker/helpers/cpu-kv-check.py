#!/usr/bin/env python3
"""Check native CPU KV shm before loading any CUDA or model code."""
import json, math, os, shutil
GiB = 2 ** 30

def native_cpu_bytes(args, env=None):
    env = {} if env is None else env
    if str(env.get('LMCACHE_ENABLED', '')).lower() in ('1', 'true', 'yes', 'on'):
        return 0
    def value(flag):
        result = None
        for i, arg in enumerate(args):
            if arg == flag and i + 1 < len(args): result = args[i + 1]
            elif arg.startswith(flag + '='): result = arg.split('=', 1)[1]
        return result
    if 'KV_TRANSFER_CONFIG' in env:
        raw = env['KV_TRANSFER_CONFIG']
        if not raw: return 0
    elif 'CPU_KV_GIB' in env:
        gib = float(env['CPU_KV_GIB'])
        if not math.isfinite(gib) or gib < 0: raise ValueError('CPU_KV_GIB 无效')
        return math.floor(gib * GiB + .5)
    else:
        raw = value('--kv-transfer-config')
    if raw is not None:
        kv = json.loads(raw)
        extra = kv.get('kv_connector_extra_config') or {}
        if (kv.get('kv_connector', 'OffloadingConnector') != 'OffloadingConnector' or
            extra.get('spec_name', 'CPUOffloadingSpec') != 'CPUOffloadingSpec' or
            kv.get('kv_role', 'kv_both') != 'kv_both'): return 0
        count = extra.get('cpu_bytes_to_use', 8 * GiB)
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0:
            raise ValueError('CPU KV 容量无效')
        return math.floor(count + .5)
    raw = value('--kv-offloading-size')
    if raw is None or value('--kv-offloading-backend') not in (None, 'native'): return 0
    gib = float(raw)
    if not math.isfinite(gib) or gib < 0: raise ValueError('CPU KV 容量无效')
    return math.floor(gib * GiB + .5)

def check(args, env=None, usage=shutil.disk_usage):
    required = native_cpu_bytes(args, os.environ if env is None else env)
    if not required: return None
    try: space = usage('/dev/shm')
    except OSError as exc: raise ValueError('无法检查 /dev/shm；原生 CPU KV 需要容器共享内存') from exc
    if space.total <= required or space.free <= required:
        size = 2 ** math.ceil(math.log2(max(16, required / GiB + 1)))
        raise ValueError(f'CPU KV 共享内存不足：申请 {required/GiB:.3f} GiB，'
                         f'/dev/shm 总计 {space.total/GiB:.3f} GiB、可用 {space.free/GiB:.6f} GiB '
                         f'（{space.free} 字节）。请设置 --shm-size={size}g 后重建容器。')
    return {'requiredBytes':required, 'totalBytes':space.total, 'availableBytes':space.free}


def apply_cpu_environment(args, env):
    """Apply the same CPU KV override precedence as the Ultra profile importer."""
    args = list(args)
    def remove(flag):
        out = []; i = 0
        while i < len(args):
            if args[i] == flag:
                i += 1
                if i < len(args) and not args[i].startswith('--'): i += 1
            elif args[i].startswith(flag + '='): i += 1
            else: out.append(args[i]); i += 1
        return out
    if 'CPU_KV_GIB' not in env and 'KV_TRANSFER_CONFIG' not in env: return args
    for flag in ['--kv-offloading-size', '--kv-offloading-backend', '--kv-transfer-config']:
        args = remove(flag)
    if 'KV_TRANSFER_CONFIG' in env:
        if env['KV_TRANSFER_CONFIG']: args += ['--kv-transfer-config', env['KV_TRANSFER_CONFIG']]
    else:
        import re
        raw = str(env['CPU_KV_GIB'])
        if not re.fullmatch(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)', raw): raise ValueError('CPU_KV_GIB 需要非负十进制 GiB 数值')
        count = native_cpu_bytes([], env)
        if count > 2**53 - 1 or (float(raw) > 0 and count < 1): raise ValueError('CPU_KV_GIB 超出有效字节范围')
        if count:
            args += ['--kv-transfer-config', json.dumps({'kv_connector':'OffloadingConnector', 'kv_role':'kv_both',
                       'kv_connector_extra_config':{'spec_name':'CPUOffloadingSpec','cpu_bytes_to_use':count}}, separators=(',', ':'))]
    return args
