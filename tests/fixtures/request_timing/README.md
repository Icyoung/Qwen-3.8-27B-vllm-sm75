# Exact stream generator fixtures

These gzip files preserve the original UTF-8 bytes (including the Apache-2.0
license header) read from the pinned vLLM 0.30.0 cu129 image. They match commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` in vllm-project/vllm:

- `vllm/entrypoints/openai/chat_completion/serving.py`: SHA256 `ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92`
- `vllm/entrypoints/openai/completion/serving.py`: SHA256 `75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527`

The regression test applies the strict idempotent installer patch and executes
the actual generator counter/hook block without loading CUDA dependencies.
Changes to these fixtures require reviewing the new engine's streaming and
usage semantics; do not update hashes to bypass a source mismatch.
