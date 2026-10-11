"""Experimental eager PLE backend. No full table GPU allocation.

Every TP rank reads complete requested rows, so no all-reduce is required.
Returns raw FP8: the original PLE layer applies its global scale once.
CUDA Graph capture is explicitly unsupported in this first integration.
"""
import json
import os
import struct
from pathlib import Path
import torch
from torch import nn
from ple_paged_reader import PagedPleReader


class DiskPLEEmbedding(nn.Module):
    supports_prefetch = False
    is_disk_ple = True
    def dequantize(self, embeddings, output_dtype):
        return embeddings.to(output_dtype) * self.weight_scale.to(device=embeddings.device, dtype=output_dtype)

    def __init__(self, num_embeddings, embedding_dim, *, prefix, **kwargs):
        super().__init__()
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.reader = PagedPleReader(os.environ['SM75_PLE_STORE'],
                                     int(os.environ.get('SM75_PLE_CACHE_BYTES', 16777216)))
        self.shards = sorted(self.reader.tensors, key=lambda n: int(n.rsplit('shard_', 1)[1].split('.')[0]))
        if len(self.shards) != 128:
            raise ValueError('Expected 128 PLE tensor shards')
        if [int(n.rsplit('shard_', 1)[1].split('.')[0]) for n in self.shards] != list(range(128)):
            raise ValueError('Non-contiguous PLE shard IDs')
        self.shard_rows = (num_embeddings + 127)//128
        for i, name in enumerate(self.shards):
            t = self.reader.tensors[name]
            expected = min(self.shard_rows, num_embeddings-i*self.shard_rows)
            if t['shape'] != [expected, embedding_dim]:
                raise ValueError(f'Runtime/checkpoint geometry mismatch: {name}')
        model = Path(os.environ['SM75_PLE_MODEL'])
        scales = []
        for p in sorted(model.glob('model-plefp8-*.safetensors')):
            with p.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                h = json.loads(f.read(length))
                for name, t in h.items():
                    if name.endswith('ngram_embedding.weight_scale'):
                        if t['dtype'] != 'BF16' or t['shape'] != [1]:
                            raise ValueError('Unexpected global scale')
                        a,b = t['data_offsets']
                        f.seek(8+length+a)
                        raw = bytearray(f.read(b-a))
                        scales.append(torch.frombuffer(raw, dtype=torch.bfloat16).clone())
        if len(scales) != 1:
            raise ValueError('Expected one PLE global scale')
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.weight_scale = nn.Parameter(scales[0].to(self.weight.device), requires_grad=False)
        self.weight_scale.weight_loader = self._load_scale

    def _load_scale(self, param, value):
        if not torch.equal(param.detach().cpu().reshape(-1), value.detach().cpu().reshape(-1)):
            raise ValueError('Global PLE scale mismatch')

    def forward(self, ids):
        from graph_boundary import eager_output
        return eager_output(self._read_rows, ids)

    def _read_rows(self, ids):
        if ids.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Disk PLE currently requires eager mode')
        rows = ids.detach().cpu().reshape(-1).tolist()
        raw = bytearray()
        for row in rows:
            if not 0 <= row < self.org_vocab_size:
                raise IndexError('PLE ID out of range')
            shard, within = divmod(row, self.shard_rows)
            raw.extend(self.reader.rows(self.shards[shard], [within]))
        if not rows:
            return torch.empty((*ids.shape, self.embedding_dim), device=ids.device, dtype=torch.float8_e4m3fn)
        return torch.frombuffer(raw, dtype=torch.float8_e4m3fn).reshape(*ids.shape, self.embedding_dim).to(ids.device)


def embedding_factory(*args, **kwargs):
    if os.environ.get('SM75_PLE_STORE'):
        return DiskPLEEmbedding(*args, **kwargs)
    from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding
    return PLEVocabParallelEmbedding(*args, **kwargs)
