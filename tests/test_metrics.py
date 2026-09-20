import numpy as np
import pytest

from src import metrics


def test_recall_perfect():
    assert metrics.recall_at_k([1, 2, 3], {1, 2}, 3) == 1.0


def test_recall_partial():
    assert metrics.recall_at_k([1, 9, 9], {1, 2}, 3) == 0.5


def test_recall_empty_truth_nan():
    assert np.isnan(metrics.recall_at_k([1, 2], set(), 2))


def test_precision():
    assert metrics.precision_at_k([1, 2, 9], {1, 2}, 3) == pytest.approx(2 / 3)


def test_ndcg_first_position():
    # one relevant at rank 1 -> DCG = 1/log2(2) = 1 ; ideal = 1 -> ndcg 1
    assert metrics.ndcg_at_k([5, 1, 2], {5}, 3) == pytest.approx(1.0)


def test_ndcg_second_position():
    # relevant at rank 2 -> 1/log2(3) ~= 0.6309
    assert metrics.ndcg_at_k([9, 5], {5}, 2) == pytest.approx(1 / np.log2(3))


def test_hit_rate():
    assert metrics.hit_rate_at_k([1, 2], {2}, 2) == 1.0
    assert metrics.hit_rate_at_k([1, 2], {7}, 2) == 0.0


def test_coverage():
    assert metrics.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)


def test_ild_identical_items_low_diversity():
    emb = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    assert metrics.intra_list_diversity([0, 1, 2], emb) == pytest.approx(0.0)


def test_ild_orthogonal_high_diversity():
    emb = np.eye(3, dtype=np.float32)
    assert metrics.intra_list_diversity([0, 1, 2], emb) == pytest.approx(1.0)
