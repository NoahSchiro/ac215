from pathlib import Path

from src.data.gazecapture import GazeCapture


def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "GazeCapture"

    data = GazeCapture("train", data_root)

    x_avg, y_avg = 0, 0
    for i in range(100):
        x_avg += data[i]["label"][0]
        y_avg += data[i]["label"][1]

    print(x_avg/100)
    print(y_avg/100)

if __name__ == "__main__":
    main()
