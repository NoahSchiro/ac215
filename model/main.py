from pathlib import Path

from src.data.mpiifacegaze import parse_mpii
from src.data.schema import write_manifest
from src.data.utils import filter_dataset

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "MPIIFaceGaze"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "mpiifacegaze"
    / "filtered_manifest.json"
)


def main() -> None:
    print(f"loading MPIIFaceGaze from {DATA_ROOT}")
    samples = parse_mpii(DATA_ROOT)

    print(f"filtering {len(samples)} samples")
    good = filter_dataset(samples)

    write_manifest(good, OUTPUT_PATH)
    print(f"wrote {len(good)} good samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
