#!/usr/bin/env python3
"""Load the private native service config without exposing API keys in argv."""

import json, os, runpy, stat, sys
from pathlib import Path

p = Path(sys.argv[1])
info = p.lstat()
if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
    raise RuntimeError("Private service config ownership/permissions mismatch")
config = json.loads(p.read_text())
os.environ.clear()
os.environ.update(config["env"])
sys.argv = ["vllm", "serve", *config["args"], "--api-key", os.environ["VLLM_API_KEY"]]
runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")
