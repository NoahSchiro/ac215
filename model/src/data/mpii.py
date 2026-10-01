import os
from typing import Any

from torch.utils.data import Dataset


class MPII(Dataset):
    def __init__(self, dir: os.PathLike[str]) -> None: ...
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError
