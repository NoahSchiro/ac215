from pathlib import Path

from src.data.mpiifacegaze import MPII

def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "MPIIFaceGaze"

    train = MPII(list(range(0, 13)), data_root)
    val = MPII(14, data_root)

    print(train[0].keys())
    print(val[0].keys())

if __name__ == "__main__":
    main()
