from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def sample_csv_paths(repository_root: Path) -> tuple[Path, Path]:
    soil = repository_root / "data" / "soil.csv"
    yield_data = repository_root / "data" / "yield.csv"
    assert soil.is_file()
    assert yield_data.is_file()
    return soil, yield_data


@pytest.fixture
def separated_matrix() -> np.ndarray:
    return np.array(
        [
            [-5.2, -5.0],
            [-4.9, -5.1],
            [-5.0, -4.8],
            [0.0, 0.2],
            [0.2, -0.1],
            [-0.2, 0.0],
            [5.0, 5.1],
            [5.2, 4.9],
            [4.8, 5.0],
        ],
        dtype=float,
    )


def same_partition(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare cluster memberships without depending on numeric label names."""

    left = np.asarray(left)
    right = np.asarray(right)
    return bool(np.array_equal(left[:, None] == left[None, :], right[:, None] == right[None, :]))
