# SPDX-License-Identifier: Apache-2.0
"""Attach exact active usage to the reviewed vLLM 0.30 streaming generators.

No API schema, request, SSE body or generator count is changed. The metadata is
already shared through raw_request.state; Pydantic does not serialize this
undeclared internal dictionary entry. Refuse unknown source and partial edits.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import stat
import uuid

PATCHES = {
    "chat_completion": ("ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92",
                        "num_choices == 1", "num_prompt_tokens"),
    "completion": ("75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527",
                   "num_choices == 1 and num_prompts == 1", "num_prompt_tokens[0]"),
}
ANCHOR = "                    previous_num_tokens[i] += len(output.token_ids)\n"


def usage_block(kind):
    _, guard, prompt = PATCHES[kind]
    return (
        "                    # vllm-sm75: exact active usage; never added to the SSE payload.\n"
        f"                    if {guard}:\n"
        "                        request_metadata.__dict__[\"_sm75_stream_usage\"] = {\n"
        f"                            \"prompt_tokens\": {prompt},\n"
        "                            \"completion_tokens\": previous_num_tokens[i],\n"
        "                        }\n"
    )


def patch_source(data, kind):
    expected, _, _ = PATCHES[kind]
    text = data.decode("utf-8")
    block = usage_block(kind)
    if block in text:
        if text.count(block) != 1:
            raise RuntimeError(f"Duplicate SM75 stream usage hook in {kind}")
        original = text.replace(block, "", 1)
        if hashlib.sha256(original.encode()).hexdigest() != expected:
            raise RuntimeError(f"Unknown or partially patched vLLM 0.30 {kind} source")
        return data
    if hashlib.sha256(data).hexdigest() != expected or text.count(ANCHOR) != 1:
        raise RuntimeError(f"Unknown vLLM 0.30 {kind} source; exact reviewed source is required")
    patched = text.replace(ANCHOR, ANCHOR + block, 1)
    ast.parse(patched)
    return patched.encode()


def patch_package(package_root):
    changes = []
    # Validate both sources before any mutation; upgrades/custom edits fail closed.
    for kind in PATCHES:
        file = Path(package_root) / f"entrypoints/openai/{kind}/serving.py"
        original = file.read_bytes()
        updated = patch_source(original, kind)
        if updated != original:
            changes.append((file, updated))
    for file, data in changes:
        temporary = file.with_name(f".sm75-stream-usage-{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as output:
                output.write(data)
            temporary.chmod(stat.S_IMODE(file.stat().st_mode))
            os.replace(temporary, file)
        finally:
            temporary.unlink(missing_ok=True)
    return len(changes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    print(f"Patched {patch_package(parser.parse_args().package_root)} stream usage sources")
