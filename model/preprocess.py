"""Run a small, reproducible MPIIFaceGaze preprocessing demonstration."""

import hashlib
import json
import os
from pathlib import Path

from src.data.mpiifacegaze import parse_mpii_sample
from src.data.schema import write_manifest
from src.data.utils import build_webdataset


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/MPIIFaceGaze"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/artifacts"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "5"))


def sha256_file(path: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if MAX_SAMPLES < 1:
        raise ValueError("MAX_SAMPLES must be positive")

    subject_dir = DATA_ROOT / "p00"
    annotation_path = subject_dir / "p00.txt"

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotations: {annotation_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Select a small number of valid records in deterministic order.
    samples = []

    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        sample = parse_mpii_sample(subject_dir, line)

        if sample is not None:
            samples.append(sample)

        if len(samples) >= MAX_SAMPLES:
            break

    if not samples:
        raise RuntimeError("No valid input samples found")

    samples.sort(key=lambda sample: sample.sample_id)

    # Create the deterministic input manifest.
    manifest_path = OUTPUT_ROOT / "mpii_filtered.json"
    write_manifest(samples, manifest_path)

    manifest_hash = sha256_file(manifest_path)

    print(f"Selected {len(samples)} input samples")
    print(f"Manifest SHA256: {manifest_hash}", flush=True)

    # Run the team's existing preprocessing implementation.
    shards_dir = OUTPUT_ROOT / "shards"
    build_webdataset(samples, shards_dir, num_threads=1)

    shard_files = sorted(shards_dir.glob("*.tar"))

    if not shard_files:
        raise RuntimeError("Preprocessing produced no shard files")

    # Record the manifest hash and each output shard hash.
    index = {
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_hash,
        "input_sample_count": len(samples),
        "shards": [
            {
                "filename": shard.name,
                "sha256": sha256_file(shard),
            }
            for shard in shard_files
        ],
    }

    index_path = OUTPUT_ROOT / "index.json"

    with index_path.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2)
        file.write("\n")

    print(f"Created {index_path}", flush=True)
    print("Small-sample preprocessing completed.", flush=True)


if __name__ == "__main__":
    main()
