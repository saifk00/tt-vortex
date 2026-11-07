"""Fetch the pinned LLVM development SDK; never build LLVM from source."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    # LLVM_SYSPATH can also name an existing, caller-provided SDK.
    if (destination / "bin/mlir-tblgen").is_file():
        return
    if destination.exists():
        parser.error(f"{destination} exists but does not contain bin/mlir-tblgen")
    spec = json.loads(Path(__file__).with_name("dependencies.lock.json").read_text())["triton"]["llvm"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="llvm-download-") as directory:
        work = Path(directory)
        archive = work / "llvm.tar.gz"
        print(f"Downloading {spec['url']}", flush=True)
        with urllib.request.urlopen(spec["url"], timeout=60) as source, archive.open("wb") as output:
            shutil.copyfileobj(source, output)
        with archive.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != spec["sha256"]:
                raise RuntimeError("LLVM archive SHA-256 does not match dependencies.lock.json")
        extracted = work / "extracted"
        with tarfile.open(archive) as source:
            source.extractall(extracted, filter="data")
        tablegen, = extracted.glob("*/bin/mlir-tblgen")
        tablegen.parent.parent.rename(destination)
    print(f"LLVM SDK ready: {destination}")


if __name__ == "__main__":
    main()
