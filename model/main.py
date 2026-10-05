from pathlib import Path

from src.data.mpiifacegaze import MPII
from src.data.gazecapture import parse_gazecapture


def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "GazeCapture"

    data = parse_gazecapture(data_root)

    print(len(data))
    print(data[0])

if __name__ == "__main__":
    main()
