from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from armbench.seeds import (
    DEFAULT_SEEDS_FILE,
    LockedSeedError,
    SeedSplit,
    check_seeds_allowed,
    load_seed_split,
)


@pytest.fixture(scope="module")
def split() -> SeedSplit:
    return load_seed_split(DEFAULT_SEEDS_FILE)


def test_default_split_matches_plan(split: SeedSplit) -> None:
    assert split.splits["dev"].seeds == range(0, 10)
    assert split.splits["skill_validation"].seeds == range(20, 40)
    assert split.splits["final_eval"].seeds == range(100, 120)
    assert split.splits["final_eval"].locked
    assert not split.splits["dev"].locked
    assert not split.splits["skill_validation"].locked


def test_split_of(split: SeedSplit) -> None:
    assert split.split_of(0) == "dev"
    assert split.split_of(39) == "skill_validation"
    assert split.split_of(100) == "final_eval"
    assert split.split_of(50) is None


def test_overlapping_ranges_rejected() -> None:
    raw = {
        "schema_version": 1,
        "splits": {
            "a": {"description": "", "start": 0, "end": 10},
            "b": {"description": "", "start": 10, "end": 20},
        },
    }
    with pytest.raises(ValueError, match="overlap"):
        SeedSplit.model_validate(raw)


def test_inverted_range_rejected() -> None:
    raw = {"schema_version": 1, "splits": {"a": {"description": "", "start": 5, "end": 1}}}
    with pytest.raises(ValueError, match="end"):
        SeedSplit.model_validate(raw)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 1, "splits": {}, "extra": 1}))
    with pytest.raises(ValueError, match="extra"):
        load_seed_split(p)


def test_locked_seed_rejected_by_default(split: SeedSplit) -> None:
    with pytest.raises(LockedSeedError, match=r"\[100, 119\]"):
        check_seeds_allowed([0, 100, 119], split)


def test_final_eval_flag_alone_is_not_enough(split: SeedSplit) -> None:
    with pytest.raises(LockedSeedError):
        check_seeds_allowed([100], split, final_eval=True)
    with pytest.raises(LockedSeedError):
        check_seeds_allowed([100], split, final_eval=True, protocol_hash="")


def test_final_eval_with_hash_allows(split: SeedSplit) -> None:
    check_seeds_allowed(list(range(100, 120)), split, final_eval=True, protocol_hash="abc123")


@given(st.lists(st.integers(min_value=0, max_value=99)))
def test_unlocked_seeds_always_allowed(seeds: list[int]) -> None:
    check_seeds_allowed(seeds, load_seed_split(DEFAULT_SEEDS_FILE))


@given(st.lists(st.integers(min_value=100, max_value=119), min_size=1))
def test_any_locked_seed_blocks(seeds: list[int]) -> None:
    with pytest.raises(LockedSeedError):
        check_seeds_allowed(seeds, load_seed_split(DEFAULT_SEEDS_FILE))
