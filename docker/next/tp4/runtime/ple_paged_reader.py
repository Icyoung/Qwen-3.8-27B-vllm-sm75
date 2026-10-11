"""Synchronous PLE row reader for the actual tensor-aligned page manifest.

Raw FP8 bytes only. Scaling, tensor placement and vLLM integration are separate.
Handles 160-byte rows crossing power-of-two page boundaries.
"""
import json
import os
import time
from collections import OrderedDict
from pathlib import Path


class PagedPleReader:
    def __init__(self, root, cache_bytes):
        self.root = Path(root)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        self.page_bytes = manifest['page_bytes']
        if self.page_bytes <= 0 or cache_bytes < 0:
            raise ValueError('Invalid page/cache size')
        self.size = (self.root / 'pages.bin').stat().st_size
        if manifest['bytes'] != self.size:
            raise ValueError('Incomplete store')
        self.tensors = {t['name']: t for t in manifest['tensors']}
        for t in self.tensors.values():
            if 'ple_embedding' not in t['name'] or '.experts.' in t['name']:
                raise ValueError('Non-PLE tensor in store')
            if t['offset'] < 0 or t['bytes'] != t['shape'][0]*t['row_bytes'] or t['offset']+t['bytes'] > self.size:
                raise ValueError('Invalid tensor bounds')
        self.fd = os.open(self.root / 'pages.bin', os.O_RDONLY)
        self.keep_os_cache = os.environ.get('SM75_PLE_KEEP_OS_CACHE', '0') == '1'
        self.capacity = cache_bytes // self.page_bytes
        self.cache = OrderedDict()
        self.hits = self.misses = self.io_ns = 0

    def _page(self, page):
        if page in self.cache:
            self.hits += 1
            self.cache.move_to_end(page)
            return self.cache[page]
        start = time.perf_counter_ns()
        data = os.pread(self.fd, self.page_bytes, page * self.page_bytes)
        self.io_ns += time.perf_counter_ns() - start
        self.misses += 1
        if len(data) != self.page_bytes:
            raise IOError('Short PLE page')
        if not self.keep_os_cache and hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(self.fd, page * self.page_bytes, self.page_bytes, os.POSIX_FADV_DONTNEED)
        if self.capacity:
            self.cache[page] = data
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return data

    def rows(self, tensor, ids):
        t = self.tensors[tensor]
        result = bytearray()
        for row in ids:
            if type(row) is not int or not 0 <= row < t['shape'][0]:
                raise IndexError('PLE row out of bounds')
            at = t['offset'] + row*t['row_bytes']
            remaining = t['row_bytes']
            while remaining:
                page, offset = divmod(at, self.page_bytes)
                count = min(remaining, self.page_bytes-offset)
                result.extend(self._page(page)[offset:offset+count])
                remaining -= count
                at += count
        return bytes(result)

    def close(self):
        os.close(self.fd)
