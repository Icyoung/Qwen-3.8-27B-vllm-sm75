"""Stream committed standard-only sources to a portable build context."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root):
    raise SystemExit("Commit tracked changes before packaging")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
data = subprocess.check_output(["git", "show", f"{revision}:docker/Dockerfile"], cwd=root)
output = Path(sys.argv[1])
with subprocess.Popen(["git", "archive", "--format=tar", revision], cwd=root, stdout=subprocess.PIPE) as process:
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|" ) as source, tarfile.open(output, "w") as dest:
            for entry in source:
                if entry.name.startswith(("evidence/", ".git/", "ultra/")):
                    raise SystemExit("Unexpected private or removed build input: " + entry.name)
                dest.addfile(entry, source.extractfile(entry) if entry.isfile() else None)
            entry = tarfile.TarInfo("Dockerfile.full")
            entry.size = len(data)
            entry.mode = 0o644
            dest.addfile(entry, io.BytesIO(data))
        if process.wait() != 0:
            raise RuntimeError("git archive failed")
    except BaseException:
        process.terminate()
        output.unlink(missing_ok=True)
        raise
sha = hashlib.sha256()
with output.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(chunk)
manifest = {"sourceRevision": revision, "edition": "standard", "target": "final",
            "archiveSha256": sha.hexdigest(), "dockerfileSha256": hashlib.sha256(data).hexdigest()}
output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
