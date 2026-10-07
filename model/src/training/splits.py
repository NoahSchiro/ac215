"""Participant-disjoint splits.

MPIIFaceGaze has 15 subjects and ~2.5k images each, so the thing that limits
generalization is identity count, not image count. Any split that puts the
same participant in both train and eval reports a number that will not
survive a new user, which is the number we actually care about.
"""

from dataclasses import dataclass

NUM_PARTICIPANTS = 15
ALL_PARTICIPANTS = tuple(range(NUM_PARTICIPANTS))


@dataclass(frozen=True, slots=True)
class Split:
    """Participant ids for one experiment. Guaranteed pairwise disjoint."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        groups = {"train": self.train, "val": self.val, "test": self.test}
        for name, ids in groups.items():
            for participant in ids:
                if not 0 <= participant < NUM_PARTICIPANTS:
                    raise ValueError(
                        f"{name} participant {participant} is out of range"
                        f" (0..{NUM_PARTICIPANTS - 1})"
                    )
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} contains duplicate participants: {ids}")
        if not self.train:
            raise ValueError("train split must be non-empty")
        if not self.val:
            raise ValueError("val split must be non-empty")

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                raise ValueError(
                    f"{left} and {right} share participants {sorted(overlap)};"
                    " splits must be participant-disjoint"
                )

    @property
    def held_out(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.val) | set(self.test)))


def holdout_split(val: int | list[int], test: int | list[int] | None = None) -> Split:
    """Everyone not named becomes train. Use this for day-to-day iteration."""
    val_ids = (val,) if isinstance(val, int) else tuple(val)
    if test is None:
        test_ids: tuple[int, ...] = ()
    else:
        test_ids = (test,) if isinstance(test, int) else tuple(test)

    excluded = set(val_ids) | set(test_ids)
    train_ids = tuple(p for p in ALL_PARTICIPANTS if p not in excluded)
    return Split(train=train_ids, val=val_ids, test=test_ids)


def leave_one_out_folds() -> list[Split]:
    """The 15 folds of the standard MPIIFaceGaze protocol.

    This is the comparable-to-published number, but it is 15 training runs.
    Use `holdout_split` while iterating and save this for the final report.
    """
    return [holdout_split(val=participant) for participant in ALL_PARTICIPANTS]
