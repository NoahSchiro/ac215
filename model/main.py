from pathlib import Path
from time import time

import numpy as np

from src.data.mpiifacegaze import MPII


def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "MPIIFaceGaze"

    data = MPII(list(range(0, 13)), data_root)

    start = time()
    for i in range(1000):
        _ = data[i]
    end = time()

    print(f"Sequential index access: {(end-start)} ms per sample")

    start = time()
    for _ in range(1000):
        idx = int(np.random.uniform(low=0, high=len(data)))
        _ = data[idx]
    end = time()

    print(f"Random index access: {(end-start)} ms")

if __name__ == "__main__":
    main()
